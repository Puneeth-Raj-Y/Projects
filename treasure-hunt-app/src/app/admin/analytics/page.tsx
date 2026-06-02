'use client'

import { useEffect, useState } from 'react'
import {
  BarChart3, Users, MapPin, ClipboardCheck, TrendingUp,
  CheckCircle, Clock, RefreshCw, Trophy
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/hooks/use-toast'
import { formatTime } from '@/lib/utils'

export default function AdminAnalyticsPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  async function fetchAnalytics() {
    setLoading(true)
    try {
      const res = await fetch('/api/admin/analytics')
      const json = await res.json()
      if (json.success) setData(json.data)
    } catch {
      toast({ title: 'Error fetching analytics', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalytics()
  }, [])

  if (loading) {
    return (
      <div className="flex flex-col gap-6">
        <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-primary" /> Analytics
        </h1>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Card key={i} className="glass"><CardContent className="p-6 h-28 shimmer rounded-xl" /></Card>
          ))}
        </div>
      </div>
    )
  }

  const { overview, topTeams } = data

  const statCards = [
    {
      label: 'Total Teams',
      value: overview.totalTeams,
      sub: `${overview.activeTeams} active`,
      icon: Users,
      color: 'text-blue-500',
      bg: 'bg-blue-500/10',
    },
    {
      label: 'Locations',
      value: overview.totalLocations,
      sub: 'Hunt checkpoints',
      icon: MapPin,
      color: 'text-emerald-500',
      bg: 'bg-emerald-500/10',
    },
    {
      label: 'Submissions',
      value: overview.totalSubmissions,
      sub: `${overview.pendingSubmissions} pending review`,
      icon: ClipboardCheck,
      color: 'text-amber-500',
      bg: 'bg-amber-500/10',
    },
    {
      label: 'Completed Teams',
      value: overview.completedTeams,
      sub: `${overview.completionRate}% completion rate`,
      icon: Trophy,
      color: 'text-purple-500',
      bg: 'bg-purple-500/10',
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary" />
            Analytics
          </h1>
          <p className="text-muted-foreground text-sm">Event performance overview and team statistics.</p>
        </div>
        <Button variant="outline" onClick={fetchAnalytics} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statCards.map((stat) => (
          <Card key={stat.label} className="glass card-hover">
            <CardContent className="p-6">
              <div className="flex items-center justify-between pb-2">
                <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                <div className={`w-8 h-8 rounded-lg ${stat.bg} flex items-center justify-center`}>
                  <stat.icon className={`h-4 w-4 ${stat.color}`} />
                </div>
              </div>
              <div className="text-3xl font-black">{stat.value}</div>
              <p className="text-xs text-muted-foreground mt-1">{stat.sub}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Completion Overview */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-500" />
              Hunt Progress
            </CardTitle>
            <CardDescription>Overall event completion stats</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div>
              <div className="flex justify-between text-sm mb-2 font-medium">
                <span>Team Completion Rate</span>
                <span className="text-emerald-500">{overview.completionRate}%</span>
              </div>
              <Progress value={overview.completionRate} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                {overview.completedTeams} of {overview.totalTeams} teams finished
              </p>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2 font-medium">
                <span>Submission Approval Rate</span>
                <span className="text-primary">
                  {overview.totalSubmissions > 0
                    ? Math.round((overview.approvedSubmissions / overview.totalSubmissions) * 100)
                    : 0}%
                </span>
              </div>
              <Progress
                value={overview.totalSubmissions > 0
                  ? Math.round((overview.approvedSubmissions / overview.totalSubmissions) * 100)
                  : 0}
                className="h-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                {overview.approvedSubmissions} approved, {overview.pendingSubmissions} pending
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-card border border-border/50 text-center">
                <Clock className="w-5 h-5 text-purple-500 mx-auto mb-1" />
                <div className="text-2xl font-black">{formatTime(overview.avgCompletionMinutes)}</div>
                <div className="text-xs text-muted-foreground">Avg Completion</div>
              </div>
              <div className="p-4 rounded-xl bg-card border border-border/50 text-center">
                <CheckCircle className="w-5 h-5 text-emerald-500 mx-auto mb-1" />
                <div className="text-2xl font-black">{overview.approvedSubmissions}</div>
                <div className="text-xs text-muted-foreground">Approved Tasks</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Top Teams */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-500" />
              Top Performing Teams
            </CardTitle>
            <CardDescription>Ranked by total points</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            {topTeams.length === 0 ? (
              <div className="text-center py-10 text-muted-foreground">
                <p className="text-sm">No teams yet.</p>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {topTeams.map((team: any, i: number) => (
                  <div key={team.id} className="flex items-center gap-3 px-5 py-3 hover:bg-accent/30 transition-colors">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs flex-shrink-0 ${
                      i === 0 ? 'bg-amber-500 text-white shadow-[0_0_10px_rgba(245,158,11,0.5)]' :
                      i === 1 ? 'bg-slate-400 text-white' :
                      i === 2 ? 'bg-amber-700 text-white' :
                      'bg-muted text-muted-foreground'
                    }`}>
                      {i + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm truncate">{team.name}</div>
                      <div className="text-xs font-mono text-muted-foreground">{team.teamId}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-black text-primary">{team.totalPoints}</div>
                      {team.status === 'completed' && (
                        <Badge variant="success" className="text-[10px] px-1.5 py-0 h-4 mt-0.5">Done</Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
