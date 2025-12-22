#!/bin/bash
# Run all benchmarks and generate reports

set -e

echo "=========================================="
echo "Running Performance Benchmarks"
echo "=========================================="
echo ""

# Create benchmarks directory
mkdir -p target/benchmarks

# Run Criterion benchmarks
echo "Running Criterion benchmarks..."
cargo bench --bench extraction_benchmark -- --output-format bencher | tee target/benchmarks/criterion_results.txt

echo ""
echo "Benchmark results saved to: target/benchmarks/criterion_results.txt"
echo "HTML report available at: target/criterion/report/index.html"

# Run memory benchmarks
echo ""
echo "=========================================="
echo "Running Memory Benchmarks"
echo "=========================================="
echo ""

cargo run --release --bin memory_benchmark | tee target/benchmarks/memory_results.txt

# Run concurrency benchmarks
echo ""
echo "=========================================="
echo "Running Concurrency Benchmarks"
echo "=========================================="
echo ""

cargo run --release --bin concurrency_benchmark | tee target/benchmarks/concurrency_results.txt

# Generate summary report
echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo ""

cat target/benchmarks/criterion_results.txt | grep "time:"
echo ""
cat target/benchmarks/memory_results.txt | grep "Delta:"
echo ""
cat target/benchmarks/concurrency_results.txt | grep "Throughput:"

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo "Results saved in: target/benchmarks/"
