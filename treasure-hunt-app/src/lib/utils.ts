import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPoints(points: number): string {
  return points.toLocaleString()
}

export function formatTime(minutes: number): string {
  if (minutes < 60) return `${minutes}m`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m > 0 ? `${h}h ${m}m` : `${h}h`
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'active': return 'text-emerald-400 bg-emerald-400/10'
    case 'completed': return 'text-amber-400 bg-amber-400/10'
    case 'disqualified': return 'text-red-400 bg-red-400/10'
    case 'approved': return 'text-emerald-400 bg-emerald-400/10'
    case 'pending': return 'text-amber-400 bg-amber-400/10'
    case 'rejected': return 'text-red-400 bg-red-400/10'
    case 'resubmit': return 'text-blue-400 bg-blue-400/10'
    default: return 'text-muted-foreground bg-muted'
  }
}

export function getRankIcon(rank: number): string {
  switch (rank) {
    case 1: return '🥇'
    case 2: return '🥈'
    case 3: return '🥉'
    default: return `#${rank}`
  }
}

export function getProgressPercentage(current: number, total: number): number {
  if (total === 0) return 0
  return Math.round((current / total) * 100)
}

export function timeAgo(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const seconds = Math.floor((Date.now() - d.getTime()) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}
