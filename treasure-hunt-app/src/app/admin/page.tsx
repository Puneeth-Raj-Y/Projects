'use client'

import { useEffect, useState } from 'react'
import {
  Users, MapPin, CheckCircle, Activity, Trophy, Clock,
  ArrowRight, ShieldCheck, Zap
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/hooks/use-toast'
import Link from 'next/link'
import { formatPoints, formatTime } from '@/lib/utils'

export default function AdminDashboardPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/admin/analytics')
        const json = await res.json()
        if (json.success) setData(json.data)
      } catch {
        toast({ title: 'Error loading analytics', variant: 'destructive' })
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [toast])

  if (loading) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-2xl font-bold tracking-tight">Dashboard Overview</h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="glass">
              <CardContent className="p-6 h-28 shimmer rounded-xl" />
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const { overview, topTeams } = data

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Dashboard Overview</h1>
        <Badge variant="outline" className="px-3 py-1 text-sm bg-background/50 backdrop-blur">
          <span className="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse" />
          Live Event Active
        </Badge>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="glass card-hover">
          <CardContent className="p-6">
            <div className="flex items-center justify-between space-y-0 pb-2">
              <p className="text-sm font-medium text-muted-foreground">Total Teams</p>
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Users className="h-4 w-4 text-blue-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{overview.totalTeams}</div>
              <span className="text-xs text-muted-foreground">{overview.activeTeams} active</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass card-hover">
          <CardContent className="p-6">
            <div className="flex items-center justify-between space-y-0 pb-2">
              <p className="text-sm font-medium text-muted-foreground">Completion</p>
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <CheckCircle className="h-4 w-4 text-emerald-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{overview.completionRate}%</div>
              <span className="text-xs text-muted-foreground">{overview.completedTeams} completed</span>
            </div>
            <Progress value={overview.completionRate} className="h-1.5 mt-3" />
          </CardContent>
        </Card>

        <Card className="glass card-hover">
          <CardContent className="p-6">
            <div className="flex items-center justify-between space-y-0 pb-2">
              <p className="text-sm font-medium text-muted-foreground">Pending Review</p>
              <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <Activity className="h-4 w-4 text-amber-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{overview.pendingSubmissions}</div>
              <span className="text-xs text-muted-foreground">tasks</span>
            </div>
            {overview.pendingSubmissions > 0 && (
              <Link href="/admin/activities" className="text-xs text-amber-500 hover:underline flex items-center gap-1 mt-2">
                Review now <ArrowRight className="w-3 h-3" />
              </Link>
            )}
          </CardContent>
        </Card>

        <Card className="glass card-hover">
          <CardContent className="p-6">
            <div className="flex items-center justify-between space-y-0 pb-2">
              <p className="text-sm font-medium text-muted-foreground">Avg Time</p>
              <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <Clock className="h-4 w-4 text-purple-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold">{formatTime(overview.avgCompletionMinutes)}</div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Leaderboard Snapshot */}
        <Card className="glass flex flex-col h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-500" />
              Top Teams
            </CardTitle>
            <CardDescription>Current points leaders</CardDescription>
          </CardHeader>
          <CardContent className="flex-1">
            <div className="space-y-4">
              {topTeams.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">No teams yet</div>
              ) : (
                topTeams.map((team: any, i: number) => (
                  <div key={team.id} className="flex items-center justify-between p-3 rounded-xl bg-card border border-border/50">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs ${
                        i === 0 ? 'bg-amber-500 text-white shadow-[0_0_10px_rgba(245,158,11,0.5)]' :
                        i === 1 ? 'bg-slate-400 text-white' :
                        i === 2 ? 'bg-amber-700 text-white' :
                        'bg-muted text-muted-foreground'
                      }`}>
                        {i + 1}
                      </div>
                      <div>
                        <div className="font-semibold text-sm">{team.name}</div>
                        <div className="text-xs text-muted-foreground">{team.teamId}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-primary">{formatPoints(team.totalPoints)} pts</div>
                      {team.status === 'completed' && (
                        <Badge variant="success" className="text-[10px] px-1.5 py-0 h-4">Done</Badge>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="mt-4 pt-4 border-t border-border/50 text-center">
              <Link href="/admin/leaderboard" className="text-sm text-primary hover:underline">
                View Full Leaderboard
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card className="glass flex flex-col h-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-primary" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 grid grid-cols-2 gap-4">
            <Link href="/admin/locations">
              <div className="h-full p-4 rounded-xl border border-border/50 bg-card hover:border-primary/50 hover:bg-primary/5 transition-all flex flex-col items-center justify-center text-center gap-2 group">
                <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <MapPin className="w-5 h-5 text-blue-500" />
                </div>
                <div className="font-medium text-sm">Manage Locations</div>
              </div>
            </Link>
            <Link href="/admin/routes">
              <div className="h-full p-4 rounded-xl border border-border/50 bg-card hover:border-primary/50 hover:bg-primary/5 transition-all flex flex-col items-center justify-center text-center gap-2 group">
                <div className="w-10 h-10 rounded-full bg-purple-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <ShieldCheck className="w-5 h-5 text-purple-500" />
                </div>
                <div className="font-medium text-sm">Assign Routes</div>
              </div>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
