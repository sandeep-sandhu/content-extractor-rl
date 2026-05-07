#!/usr/bin/env python3
"""
Generate comprehensive benchmark report
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def parse_criterion_results(results_path):
    """Parse Criterion benchmark results"""
    results = {}

    with open(results_path) as f:
        for line in f:
            if 'time:' in line:
                parts = line.strip().split()
                benchmark_name = parts[0]
                time_value = float(parts[2])
                time_unit = parts[3]

                # Convert to milliseconds
                if time_unit == 'ns':
                    time_ms = time_value / 1_000_000
                elif time_unit == 'us' or time_unit == 'µs':
                    time_ms = time_value / 1000
                elif time_unit == 'ms':
                    time_ms = time_value
                else:
                    time_ms = time_value * 1000

                results[benchmark_name] = time_ms

    return results


def parse_memory_results(results_path):
    """Parse memory benchmark results"""
    results = {}

    with open(results_path) as f:
        current_benchmark = None
        for line in f:
            if ':' in line and not line.startswith(' '):
                current_benchmark = line.split(':')[0].strip()
                results[current_benchmark] = {}
            elif 'Delta:' in line and current_benchmark:
                parts = line.strip().split()
                memory_mb = float(parts[1])
                results[current_benchmark]['memory_mb'] = memory_mb

    return results


def generate_html_report(criterion_results, memory_results, output_path):
    """Generate HTML benchmark report"""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Content Extractor RL - Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .header {{
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
        }}
        .benchmark-section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            font-weight: bold;
            color: #4CAF50;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Content Extractor RL - Performance Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="benchmark-section">
        <h2>Extraction Performance</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Time (ms)</th>
                <th>Throughput</th>
            </tr>
    """

    for name, time_ms in sorted(criterion_results.items()):
        throughput = f"{1000/time_ms:.2f} ops/sec" if time_ms > 0 else "N/A"
        html += f"""
            <tr>
                <td>{name}</td>
                <td class="metric">{time_ms:.2f}</td>
                <td>{throughput}</td>
            </tr>
        """

    html += """
        </table>
    </div>
    
    <div class="benchmark-section">
        <h2>Memory Usage</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Memory Delta (MB)</th>
            </tr>
    """

    for name, data in sorted(memory_results.items()):
        if 'memory_mb' in data:
            html += f"""
            <tr>
                <td>{name}</td>
                <td class="metric">{data['memory_mb']:.2f}</td>
            </tr>
            """

    html += """
        </table>
    </div>
    
    <div class="benchmark-section">
        <h2>Summary</h2>
        <ul>
    """

    # Calculate summary statistics
    if criterion_results:
        avg_time = sum(criterion_results.values()) / len(criterion_results)
        html += f"<li>Average extraction time: <span class='metric'>{avg_time:.2f} ms</span></li>"
        if memory_results:
            total_memory = sum(
                data.get('memory_mb', 0)
                for data in memory_results.values()
            )
            html += f"<li>Total memory usage: <span class='metric'>{total_memory:.2f} MB</span></li>"
            html += """
                </ul>
            </div>
            
            <div class="footer">
                <p>Content Extractor RL - RL-based HTML Article Extraction</p>
            </div>
            </body>
            </html>
                """
            with open(output_path, 'w') as f:
                f.write(html)

            print(f"✓ HTML report generated: {output_path}")

def main():
    benchmarks_dir = Path("target/benchmarks")
    if not benchmarks_dir.exists():
        print("Error: Benchmarks directory not found. Run benchmarks first.")
    sys.exit(1)

    criterion_path = benchmarks_dir / "criterion_results.txt"
    memory_path = benchmarks_dir / "memory_results.txt"

    criterion_results = {}
    memory_results = {}

    if criterion_path.exists():
        criterion_results = parse_criterion_results(criterion_path)
        print(f"✓ Parsed Criterion results: {len(criterion_results)} benchmarks")

    if memory_path.exists():
        memory_results = parse_memory_results(memory_path)
        print(f"✓ Parsed memory results: {len(memory_results)} benchmarks")

    # Generate HTML report
    output_path = benchmarks_dir / "benchmark_report.html"
    generate_html_report(criterion_results, memory_results, output_path)

    # Generate JSON report
    json_output = {
        'generated_at': datetime.now().isoformat(),
        'criterion_benchmarks': criterion_results,
        'memory_benchmarks': memory_results,
    }

    json_path = benchmarks_dir / "benchmark_report.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"✓ JSON report generated: {json_path}")
    print(f"\nOpen {output_path} in your browser to view the report.")


if __name__ == '__main__':
    main()
