// Costs per request are tiny, so show enough digits to be meaningful
export function formatCost(dollars: number | null): string {
    if (dollars === null) return "-"
    if (dollars === 0) return "$0"
    if (dollars < 0.0001) return `$${dollars.toFixed(6)}`
    if (dollars < 0.01) return `$${dollars.toFixed(5)}`
    return `$${dollars.toFixed(2)}`
}

export function formatPercent(value: number | null): string {
    if (value === null) return "-"
    return `${Math.round(value * 100)}%`
}