import benchmarkData from "@/content/benchmark.json"
import type { Benchmark } from "@/lib/dashboard"

// The exported benchmark (eval/export_benchmark.py). Every benchmark number on the site comes from here.
export const benchmark = benchmarkData as unknown as Benchmark

export type CurvePoint = Benchmark["curve"]["points"][number]

// The curve point for a quality setting (tau), e.g. 0.86 -> the Balanced result
export function curveAt(tau: number): CurvePoint {
  const points = benchmark.curve.points
  return points.reduce((best, p) => (Math.abs(p.tau - tau) < Math.abs(best.tau - tau) ? p : best), points[0])
}
