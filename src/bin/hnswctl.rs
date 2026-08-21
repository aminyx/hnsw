//! `hnswctl` — build, query, benchmark and measure an HNSW index.
//!
//! No argument-parsing dependency: the surface is a handful of `--key value`
//! pairs, and hand-rolling it keeps the crate's dependency list genuinely
//! empty rather than nearly empty.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use hnsw::brute::ground_truth;
use hnsw::dataset::Dataset;
use hnsw::distance::Metric;
use hnsw::error::Error;
use hnsw::eval::{brute_force_qps, markdown_table, recall_sweep, text_table};
use hnsw::index::{Hnsw, Params, Searcher, Selection};
use hnsw::BruteForce;

const USAGE: &str = "\
hnswctl — Hierarchical Navigable Small World index

USAGE
  hnswctl <command> [options]

COMMANDS
  gen       write a synthetic dataset to a file
  build     build an index from a vector file and save it
  query     search a saved index
  info      print the structure of a saved index
  bench     time build and query throughput on generated data
  recall    sweep ef_search and report recall@k against an exact scan

DATA OPTIONS (gen, build, query --input)
  --format csv|f32     csv: one comma-separated vector per line (default)
                       f32: raw little-endian f32, needs --dims
  --dims N             vector width; inferred from the first csv row
  --ids                treat the first csv column as a u64 id

INDEX OPTIONS (build, bench, recall)
  --metric cosine|dot|l2     default cosine
  --m N                      links per node per layer (default 16)
  --m-max0 N                 layer-0 degree cap (default 2*m)
  --ef-construction N        build-time beam width (default 200)
  --ef-search N              default query beam width (default 64)
  --ml F                     level distribution scale (default 1/ln(m))
  --seed N                   PRNG seed
  --selection heuristic|naive
                             neighbour selection; 'naive' is the broken
                             top-M baseline, kept for comparison

EXAMPLES
  hnswctl gen --out data.csv --n 20000 --dims 64 --clusters 24
  hnswctl build --input data.csv --out data.hnsw --metric l2
  hnswctl query --index data.hnsw --vector 0.1,0.2,0.3 --k 10
  hnswctl recall --n 20000 --dims 64 --k 10 --ef 8,16,32,64,128,256 --markdown
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let command = args[0].clone();
    let opts = match Options::parse(&args[1..]) {
        Ok(o) => o,
        Err(message) => return fail(&message),
    };

    let result = match command.as_str() {
        "gen" => cmd_gen(&opts),
        "build" => cmd_build(&opts),
        "query" => cmd_query(&opts),
        "info" => cmd_info(&opts),
        "bench" => cmd_bench(&opts),
        "recall" => cmd_recall(&opts),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => fail(&message),
    }
}

fn fail(message: &str) -> ExitCode {
    let mut err = io::stderr().lock();
    let _ = writeln!(err, "hnswctl: {message}");
    ExitCode::FAILURE
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Options {
    values: BTreeMap<String, String>,
    flags: Vec<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        const FLAGS: [&str; 3] = ["ids", "markdown", "verify"];
        let mut opts = Options::default();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            let Some(key) = arg.strip_prefix("--") else {
                return Err(format!(
                    "unexpected argument {arg:?}; options start with --"
                ));
            };
            if FLAGS.contains(&key) {
                opts.flags.push(key.to_string());
                i += 1;
                continue;
            }
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("--{key} needs a value"))?;
            opts.values.insert(key.to_string(), value.clone());
            i += 2;
        }
        Ok(opts)
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    fn text(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.text(name)
            .ok_or_else(|| format!("--{name} is required"))
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.text(name) {
            None => Ok(default),
            Some(raw) => raw
                .parse()
                .map_err(|_| format!("--{name} expects a number, got {raw:?}")),
        }
    }

    fn params(&self) -> Result<Params, String> {
        let m = self.number("m", 16usize)?;
        let mut params = Params::default().with_m(m);
        params.m_max0 = self.number("m-max0", params.m_max0)?;
        params.ef_construction = self.number("ef-construction", params.ef_construction)?;
        params.ef_search = self.number("ef-search", params.ef_search)?;
        params.ml = self.number("ml", params.ml)?;
        params.seed = self.number("seed", params.seed)?;
        params.selection = match self.text("selection") {
            None | Some("heuristic") => Selection::Heuristic,
            Some("naive" | "naive-top-m") => Selection::NaiveTopM,
            Some(other) => return Err(format!("unknown --selection {other:?}")),
        };
        params.validate().map_err(|e| e.to_string())?;
        Ok(params)
    }

    fn metric(&self) -> Result<Metric, String> {
        match self.text("metric") {
            None => Ok(Metric::Cosine),
            Some(raw) => Metric::parse(raw).ok_or_else(|| format!("unknown --metric {raw:?}")),
        }
    }

    fn ef_list(&self) -> Result<Vec<usize>, String> {
        let raw = self.text("ef").unwrap_or("8,16,32,64,128,256");
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| format!("--ef expects a comma-separated list, got {s:?}"))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Vector files
// ---------------------------------------------------------------------------

struct Vectors {
    dims: usize,
    ids: Option<Vec<u64>>,
    flat: Vec<f32>,
}

impl Vectors {
    fn len(&self) -> usize {
        self.flat.len() / self.dims.max(1)
    }

    fn id(&self, i: usize) -> u64 {
        self.ids.as_ref().map_or(i as u64, |ids| ids[i])
    }

    fn get(&self, i: usize) -> &[f32] {
        &self.flat[i * self.dims..(i + 1) * self.dims]
    }
}

fn read_vectors(path: &str, opts: &Options) -> Result<Vectors, String> {
    let format = opts.text("format").unwrap_or("csv");
    match format {
        "csv" => read_csv(path, opts.flag("ids")),
        "f32" | "bin" | "binary" => {
            let dims = opts.number::<usize>("dims", 0)?;
            if dims == 0 {
                return Err("--dims is required for --format f32".into());
            }
            read_f32(path, dims)
        }
        other => Err(format!("unknown --format {other:?}")),
    }
}

fn read_csv(path: &str, has_ids: bool) -> Result<Vectors, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut dims = 0usize;
    let mut flat = Vec::new();
    let mut ids = has_ids.then(Vec::new);

    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split(',').map(str::trim);
        if let Some(ids) = ids.as_mut() {
            let raw = fields.next().unwrap_or("");
            let id = raw
                .parse::<u64>()
                .map_err(|_| format!("{path}:{}: {raw:?} is not a u64 id", lineno + 1))?;
            ids.push(id);
        }
        let before = flat.len();
        for field in fields {
            let value = field
                .parse::<f32>()
                .map_err(|_| format!("{path}:{}: {field:?} is not a number", lineno + 1))?;
            flat.push(value);
        }
        let width = flat.len() - before;
        if dims == 0 {
            dims = width;
        } else if width != dims {
            return Err(format!(
                "{path}:{}: row has {width} values, expected {dims}",
                lineno + 1
            ));
        }
    }
    if dims == 0 {
        return Err(format!("{path} contains no vectors"));
    }
    Ok(Vectors { dims, ids, flat })
}

fn read_f32(path: &str, dims: usize) -> Result<Vectors, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path} is not a whole number of f32 values"));
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if flat.len() % dims != 0 {
        return Err(format!(
            "{path} holds {} floats, not a multiple of --dims {dims}",
            flat.len()
        ));
    }
    Ok(Vectors {
        dims,
        ids: None,
        flat,
    })
}

fn write_csv(path: &str, data: &Dataset) -> Result<(), String> {
    let mut out = String::with_capacity(data.len() * data.dims() * 8);
    for vector in data {
        for (i, value) in vector.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{value:.6}");
        }
        out.push('\n');
    }
    fs::write(path, out).map_err(|e| format!("cannot write {path}: {e}"))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_gen(opts: &Options) -> Result<(), String> {
    let out = opts.required("out")?;
    let n = opts.number("n", 10_000usize)?;
    let dims = opts.number("dims", 64usize)?;
    let clusters = opts.number("clusters", 16usize)?;
    let spread = opts.number("spread", 0.22f32)?;
    let seed = opts.number("seed", 1_234_567u64)?;

    let data = Dataset::clustered(n, dims, clusters, spread, seed);
    write_csv(out, &data)?;
    println!("wrote {n} x {dims} vectors in {clusters} clusters to {out}");
    Ok(())
}

fn cmd_build(opts: &Options) -> Result<(), String> {
    let input = opts.required("input")?;
    let out = opts.required("out")?;
    let metric = opts.metric()?;
    let params = opts.params()?;

    let data = read_vectors(input, opts)?;
    let mut index = Hnsw::new(data.dims, metric, params).map_err(stringify)?;

    let start = Instant::now();
    for i in 0..data.len() {
        index.insert(data.id(i), data.get(i)).map_err(stringify)?;
    }
    let elapsed = start.elapsed();

    if opts.flag("verify") {
        index
            .validate()
            .map_err(|e| format!("invariant broken: {e}"))?;
    }
    index.save(out).map_err(stringify)?;
    let size = fs::metadata(out).map(|m| m.len()).unwrap_or(0);

    println!(
        "built {} vectors x {} dims in {:.2}s ({:.0}/s)",
        data.len(),
        data.dims,
        elapsed.as_secs_f64(),
        data.len() as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    println!(
        "  metric {metric}, m {}, m_max0 {}, ef_construction {}",
        params.m, params.m_max0, params.ef_construction
    );
    println!(
        "  layers {:?}, {} directed edges, mean layer-0 degree {:.1}",
        index.layer_sizes(),
        index.edge_count(),
        index.mean_degree()
    );
    println!("  saved {out} ({:.1} MiB)", size as f64 / (1024.0 * 1024.0));
    Ok(())
}

fn cmd_query(opts: &Options) -> Result<(), String> {
    let path = opts.required("index")?;
    let index = Hnsw::load(path).map_err(stringify)?;
    let k = opts.number("k", 10usize)?;
    let ef = opts.number("ef", index.params().ef_search)?;

    let queries: Vec<Vec<f32>> = if let Some(raw) = opts.text("vector") {
        let parsed: Result<Vec<f32>, String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<f32>()
                    .map_err(|_| format!("--vector: {s:?} is not a number"))
            })
            .collect();
        vec![parsed?]
    } else if let Some(input) = opts.text("input") {
        let data = read_vectors(input, opts)?;
        (0..data.len()).map(|i| data.get(i).to_vec()).collect()
    } else {
        return Err("give either --vector a,b,c or --input FILE".into());
    };

    let mut scratch = Searcher::new();
    for (qi, query) in queries.iter().enumerate() {
        if queries.len() > 1 {
            println!("query {qi}");
        }
        let (hits, stats) = index
            .search_stats(&mut scratch, query, k, ef)
            .map_err(stringify)?;
        for (rank, hit) in hits.iter().enumerate() {
            println!(
                "  {:>3}. id {:<12} distance {:.6}",
                rank + 1,
                hit.id,
                hit.distance
            );
        }
        println!(
            "  visited {} of {} nodes ({:.2}%), {} hops, ef {ef}",
            stats.visited,
            index.node_count(),
            stats.visited as f64 / index.node_count().max(1) as f64 * 100.0,
            stats.hops
        );
    }
    Ok(())
}

fn cmd_info(opts: &Options) -> Result<(), String> {
    let path = opts.required("index")?;
    let index = Hnsw::load(path).map_err(stringify)?;
    let size = fs::metadata(Path::new(path)).map(|m| m.len()).unwrap_or(0);

    println!(
        "{path} ({:.1} MiB on disk)",
        size as f64 / (1024.0 * 1024.0)
    );
    println!("  metric            {}", index.metric());
    println!("  dims              {}", index.dims());
    println!(
        "  vectors           {} live, {} tombstoned, {} nodes",
        index.len(),
        index.deleted_count(),
        index.node_count()
    );
    println!(
        "  m / m_max0        {} / {}",
        index.params().m,
        index.params().m_max0
    );
    println!("  ef_construction   {}", index.params().ef_construction);
    println!("  ef_search         {}", index.params().ef_search);
    println!("  ml                {:.6}", index.params().ml);
    println!("  seed              {}", index.params().seed);
    println!("  layers            {:?}", index.layer_sizes());
    println!("  directed edges    {}", index.edge_count());
    println!("  mean layer-0 deg  {:.2}", index.mean_degree());
    let components = index.components();
    println!(
        "  components        {} (largest {})",
        components.len(),
        components.first().copied().unwrap_or(0)
    );
    println!(
        "  reachable         {} of {} from the entry point",
        index.reachable_from_entry(0),
        index.node_count()
    );
    match index.validate() {
        Ok(()) => println!("  invariants        all hold"),
        Err(why) => println!("  invariants        BROKEN: {why}"),
    }
    Ok(())
}

struct Bench {
    index: Hnsw<Metric>,
    exact: BruteForce<Metric>,
    queries: Dataset,
    build_seconds: f64,
    n: usize,
    dims: usize,
}

fn prepare(opts: &Options) -> Result<Bench, String> {
    let n = opts.number("n", 20_000usize)?;
    let dims = opts.number("dims", 64usize)?;
    let clusters = opts.number("clusters", 24usize)?;
    let spread = opts.number("spread", 0.22f32)?;
    let query_count = opts.number("queries", 200usize)?;
    let seed = opts.number("data-seed", 20_260_821u64)?;
    let metric = opts.metric()?;
    let params = opts.params()?;

    let mut data = Dataset::clustered(n + query_count, dims, clusters, spread, seed);
    let queries = data.split_off_queries(query_count);

    let mut index = Hnsw::new(dims, metric, params).map_err(stringify)?;
    let start = Instant::now();
    index.insert_batch(0, data.flat()).map_err(stringify)?;
    let build_seconds = start.elapsed().as_secs_f64();

    let mut exact = BruteForce::new(dims, metric).map_err(stringify)?;
    exact.insert_batch(0, data.flat()).map_err(stringify)?;

    Ok(Bench {
        index,
        exact,
        queries,
        build_seconds,
        n: data.len(),
        dims,
    })
}

fn describe(bench: &Bench) {
    let index = &bench.index;
    println!(
        "index    {} vectors x {} dims, metric {}, m {}, ef_construction {}",
        bench.n,
        bench.dims,
        index.metric(),
        index.params().m,
        index.params().ef_construction
    );
    println!(
        "build    {:.2}s ({:.0} vectors/s), layers {:?}, {} directed edges",
        bench.build_seconds,
        bench.n as f64 / bench.build_seconds.max(1e-9),
        index.layer_sizes(),
        index.edge_count()
    );
    let components = index.components();
    println!(
        "graph    mean layer-0 degree {:.1}, {} component{} (largest {})",
        index.mean_degree(),
        components.len(),
        if components.len() == 1 { "" } else { "s" },
        components.first().copied().unwrap_or(0)
    );
    let reachable = index.reachable_from_entry(0);
    println!(
        "reach    {reachable} of {} nodes reachable from the entry point{}",
        index.node_count(),
        if reachable == index.node_count() {
            String::new()
        } else {
            format!(" ({} stranded)", index.node_count() - reachable)
        }
    );
}

fn cmd_bench(opts: &Options) -> Result<(), String> {
    let bench = prepare(opts)?;
    let k = opts.number("k", 10usize)?;
    describe(&bench);

    let truth = ground_truth(&bench.exact, bench.queries.flat(), k).map_err(stringify)?;
    let efs = opts.ef_list()?;
    let rows =
        recall_sweep(&bench.index, bench.queries.flat(), &truth, k, &efs).map_err(stringify)?;
    let scan = brute_force_qps(&bench.exact, bench.queries.flat(), k).map_err(stringify)?;

    println!();
    print!("{}", text_table(&rows, k));
    println!();
    println!("exact scan  {scan:.0} QPS (the number to beat)");
    if let Some(best) = rows.iter().find(|r| r.recall >= 0.95) {
        println!(
            "first row at recall >= 0.95: ef_search {} -> {:.0} QPS, {:.0}x the exact scan",
            best.ef_search,
            best.qps,
            best.qps / scan.max(1e-9)
        );
    }
    Ok(())
}

fn cmd_recall(opts: &Options) -> Result<(), String> {
    let bench = prepare(opts)?;
    let k = opts.number("k", 10usize)?;
    let efs = opts.ef_list()?;

    if !opts.flag("markdown") {
        describe(&bench);
        println!();
    }
    let truth = ground_truth(&bench.exact, bench.queries.flat(), k).map_err(stringify)?;
    let rows =
        recall_sweep(&bench.index, bench.queries.flat(), &truth, k, &efs).map_err(stringify)?;

    if opts.flag("markdown") {
        let scan = brute_force_qps(&bench.exact, bench.queries.flat(), k).map_err(stringify)?;
        println!(
            "<!-- {} x {} dims, metric {}, m {}, m_max0 {}, ef_construction {}, {} queries, k={k} -->",
            bench.n,
            bench.dims,
            bench.index.metric(),
            bench.index.params().m,
            bench.index.params().m_max0,
            bench.index.params().ef_construction,
            bench.queries.len(),
        );
        println!(
            "<!-- build {:.2}s, exact scan {scan:.0} QPS -->",
            bench.build_seconds
        );
        print!("{}", markdown_table(&rows, k));
    } else {
        print!("{}", text_table(&rows, k));
    }
    Ok(())
}

fn stringify(e: Error) -> String {
    e.to_string()
}
