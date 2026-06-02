'use client'

import { useEffect, useState } from 'react'
import { Trophy, RefreshCw, Crown, CheckCircle2, Clock, MapPin } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/hooks/use-toast'

export default function AdminLeaderboardPage() {
  const [teams, setTeams] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  async function fetchLeaderboard() {
    setLoading(true)
    try {
      const res = await fetch('/api/leaderboard')
      const json = await res.json()
      if (json.success) setTeams(json.data)
    } catch {
      toast({ title: 'Error fetching leaderboard', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLeaderboard()
    // Auto-refresh every 30s
    const interval = setInterval(fetchLeaderboard, 30000)
    return () => clearInterval(interval)
  }, [])

  const getRankStyle = (rank: number) => {
    if (rank === 1) return 'bg-gradient-to-br from-amber-400 to-amber-600 text-white shadow-[0_0_20px_rgba(245,158,11,0.5)]'
    if (rank === 2) return 'bg-gradient-to-br from-slate-300 to-slate-500 text-white shadow-[0_0_15px_rgba(148,163,184,0.4)]'
    if (rank === 3) return 'bg-gradient-to-br from-amber-600 to-amber-800 text-white shadow-[0_0_15px_rgba(180,83,9,0.4)]'
    return 'bg-muted text-muted-foreground'
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Trophy className="w-6 h-6 text-amber-500" />
            Live Leaderboard
          </h1>
          <p className="text-muted-foreground text-sm">Real-time team rankings. Auto-refreshes every 30 seconds.</p>
        </div>
        <Button variant="outline" onClick={fetchLeaderboard} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Top 3 Podium */}
      {!loading && teams.length >= 3 && (
        <div className="grid grid-cols-3 gap-4 mb-2">
          {/* 2nd Place */}
          <Card className="glass text-center pt-6 pb-4 px-4 flex flex-col items-center justify-end mt-6 border-slate-400/30">
            <div className={`w-14 h-14 rounded-full flex items-center justify-center text-xl font-black mb-3 ${getRankStyle(2)}`}>2</div>
            <div className="font-bold truncate w-full">{teams[1]?.name}</div>
            <div className="text-2xl font-black text-slate-400 mt-1">{teams[1]?.totalPoints} <span className="text-xs font-normal text-muted-foreground">pts</span></div>
            {teams[1]?.isCompleted && <Badge variant="success" className="mt-2 text-[10px]">Completed</Badge>}
          </Card>

          {/* 1st Place */}
          <Card className="glass text-center py-6 px-4 flex flex-col items-center justify-end border-amber-500/40 shadow-[0_0_30px_rgba(245,158,11,0.1)] relative">
            <Crown className="w-8 h-8 text-amber-500 mb-2 drop-shadow-[0_0_8px_rgba(245,158,11,0.8)]" />
            <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-black mb-3 ${getRankStyle(1)}`}>1</div>
            <div className="font-bold text-lg truncate w-full">{teams[0]?.name}</div>
            <div className="text-3xl font-black text-amber-500 mt-1">{teams[0]?.totalPoints} <span className="text-xs font-normal text-muted-foreground">pts</span></div>
            {teams[0]?.isCompleted && <Badge variant="success" className="mt-2">Completed</Badge>}
          </Card>

          {/* 3rd Place */}
          <Card className="glass text-center pt-6 pb-4 px-4 flex flex-col items-center justify-end mt-8 border-amber-700/30">
            <div className={`w-14 h-14 rounded-full flex items-center justify-center text-xl font-black mb-3 ${getRankStyle(3)}`}>3</div>
            <div className="font-bold truncate w-full">{teams[2]?.name}</div>
            <div className="text-2xl font-black text-amber-700 mt-1">{teams[2]?.totalPoints} <span className="text-xs font-normal text-muted-foreground">pts</span></div>
            {teams[2]?.isCompleted && <Badge variant="success" className="mt-2 text-[10px]">Completed</Badge>}
          </Card>
        </div>
      )}

      {/* Full Rankings Table */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="text-lg">Full Rankings</CardTitle>
          <CardDescription>{teams.length} teams total</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="space-y-2 p-4">
              {[1,2,3,4,5].map(i => (
                <div key={i} className="h-16 rounded-xl shimmer" />
              ))}
            </div>
          ) : teams.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Trophy className="w-12 h-12 mx-auto opacity-20 mb-3" />
              <p>No teams registered yet.</p>
            </div>
          ) : (
            <div className="divide-y divide-border/50">
              {teams.map((team) => {
                const progress = team.totalLocations > 0
                  ? Math.round((team.locationsCompleted / team.totalLocations) * 100)
                  : 0

                return (
                  <div key={team.id} className={`flex items-center gap-4 p-4 hover:bg-accent/30 transition-colors ${team.rank <= 3 ? 'bg-primary/3' : ''}`}>
                    {/* Rank */}
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0 ${getRankStyle(team.rank)}`}>
                      {team.rank}
                    </div>

                    {/* Team info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold truncate">{team.name}</span>
                        {team.isCompleted && <Badge variant="success" className="text-[10px] px-1.5 py-0 h-4 flex-shrink-0"><CheckCircle2 className="w-3 h-3 mr-1" />Done</Badge>}
                        {team.status === 'disqualified' && <Badge variant="destructive" className="text-[10px] flex-shrink-0">DQ'd</Badge>}
                      </div>
                      <div className="text-xs text-muted-foreground font-mono">{team.teamId}</div>

                      {/* Progress bar */}
                      {team.totalLocations > 0 && (
                        <div className="mt-1.5 flex items-center gap-2">
                          <Progress value={progress} className="h-1 flex-1" />
                          <span className="text-[10px] text-muted-foreground flex-shrink-0 flex items-center gap-1">
                            <MapPin className="w-2.5 h-2.5" />
                            {team.locationsCompleted}/{team.totalLocations}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Points & time */}
                    <div className="text-right flex-shrink-0">
                      <div className="font-black text-xl text-primary">{team.totalPoints}</div>
                      <div className="text-xs text-muted-foreground">points</div>
                      {team.completionTime && (
                        <div className="text-xs text-emerald-500 flex items-center gap-1 justify-end mt-1">
                          <Clock className="w-3 h-3" />
                          {team.completionTime}m
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
