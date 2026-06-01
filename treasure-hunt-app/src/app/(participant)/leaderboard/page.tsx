'use client'

import { useEffect, useState } from 'react'
import { Trophy, Clock, CheckCircle2, ChevronUp } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import { getRankIcon, formatPoints, formatTime } from '@/lib/utils'

export default function LeaderboardPage() {
  const [leaderboard, setLeaderboard] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/leaderboard')
        const json = await res.json()
        if (json.success) setLeaderboard(json.data)
      } catch {
        toast({ title: 'Error loading leaderboard', variant: 'destructive' })
      } finally {
        setLoading(false)
      }
    }
    load()
    const interval = setInterval(load, 30000) // refresh every 30s
    return () => clearInterval(interval)
  }, [toast])

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto space-y-4">
        <h1 className="text-3xl font-extrabold text-center mb-8">Live Leaderboard</h1>
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-24 glass rounded-2xl shimmer" />
        ))}
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto pb-20 md:pb-0">
      <div className="text-center space-y-4 mb-10">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 glow-gold">
          <Trophy className="w-10 h-10 text-white" />
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight">Live Leaderboard</h1>
        <p className="text-muted-foreground text-lg">Top teams battling for the ultimate treasure.</p>
      </div>

      <div className="space-y-4">
        {leaderboard.length === 0 ? (
          <Card className="glass"><CardContent className="p-12 text-center text-muted-foreground">No teams active yet</CardContent></Card>
        ) : (
          leaderboard.map((team, index) => {
            const isTop3 = index < 3
            
            return (
              <div 
                key={team.id} 
                className={`relative overflow-hidden rounded-2xl p-4 sm:p-6 flex flex-col sm:flex-row sm:items-center gap-4 transition-all card-hover border
                  ${index === 0 ? 'bg-gradient-to-r from-amber-500/10 to-orange-500/10 border-amber-500/30' : 
                    index === 1 ? 'bg-slate-400/10 border-slate-400/30' : 
                    index === 2 ? 'bg-amber-700/10 border-amber-700/30' : 'bg-card border-border/50'
                  }
                `}
              >
                {/* Rank Badge */}
                <div className={`w-12 h-12 flex-shrink-0 rounded-full flex items-center justify-center text-xl font-bold shadow-sm
                  ${index === 0 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white glow-gold' : 
                    index === 1 ? 'bg-gradient-to-br from-slate-300 to-slate-500 text-white' : 
                    index === 2 ? 'bg-gradient-to-br from-amber-600 to-amber-800 text-white' : 'bg-muted text-muted-foreground'
                  }
                `}>
                  {getRankIcon(team.rank)}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-xl font-bold truncate">{team.name}</h2>
                    {team.isCompleted && <Badge variant="success" className="h-5 px-1.5"><CheckCircle2 className="w-3 h-3 mr-1"/> Done</Badge>}
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
                    <span className="font-mono">{team.teamId}</span>
                    <span className="w-1 h-1 rounded-full bg-border" />
                    <span>Locations: {team.locationsCompleted}/{team.totalLocations}</span>
                    <span className="w-1 h-1 rounded-full bg-border" />
                    <span>Activities: {team.activitiesCompleted}</span>
                  </div>
                </div>

                <div className="flex flex-row sm:flex-col items-center sm:items-end justify-between sm:justify-center gap-2 border-t sm:border-t-0 sm:border-l border-border/50 pt-4 sm:pt-0 sm:pl-6 mt-2 sm:mt-0">
                  <div className="text-center sm:text-right">
                    <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-0.5">Points</div>
                    <div className={`text-3xl font-black ${isTop3 ? 'gradient-text' : 'text-foreground'}`}>
                      {formatPoints(team.totalPoints)}
                    </div>
                  </div>
                  
                  {team.isCompleted && team.completionTime && (
                    <div className="flex items-center gap-1 text-sm text-muted-foreground bg-muted/50 px-2 py-1 rounded-md">
                      <Clock className="w-3 h-3" /> {formatTime(team.completionTime)}
                    </div>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
