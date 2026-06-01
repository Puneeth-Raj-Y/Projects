'use client'

import { useEffect, useState } from 'react'
import { Trophy, MapPin, Map, CheckCircle2, Navigation, AlertCircle, Camera, LockKeyhole, Users } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { getRankIcon, getProgressPercentage } from '@/lib/utils'
import Link from 'next/link'

export default function ParticipantDashboardPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/participant/dashboard')
        const json = await res.json()
        if (json.success) setData(json.data)
      } catch {
        toast({ title: 'Error loading dashboard', variant: 'destructive' })
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [toast])

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-32 glass rounded-2xl shimmer" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="h-64 glass rounded-2xl shimmer" />
          <div className="h-64 glass rounded-2xl shimmer" />
        </div>
      </div>
    )
  }

  const { team, rank, route, currentLocation } = data

  if (team.status === 'disqualified') {
    return (
      <Card className="border-destructive/50 bg-destructive/10">
        <CardContent className="flex flex-col items-center justify-center p-12 text-center">
          <AlertCircle className="w-16 h-16 text-destructive mb-4" />
          <h2 className="text-2xl font-bold text-destructive mb-2">Team Disqualified</h2>
          <p className="text-muted-foreground">Your team has been disqualified from the hunt. Please contact an admin for more information.</p>
        </CardContent>
      </Card>
    )
  }

  if (team.status === 'completed' || route?.isCompleted) {
    return (
      <div className="max-w-2xl mx-auto text-center space-y-6 py-12">
        <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-emerald-500/10 mb-4">
          <Trophy className="w-12 h-12 text-emerald-500" />
        </div>
        <h1 className="text-4xl font-extrabold gradient-text">Hunt Completed!</h1>
        <p className="text-xl text-muted-foreground">Congratulations {team.name}! You have successfully completed the TreasureQuest.</p>
        
        <Card className="glass mt-8 p-8 text-center">
          <div className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-2">Final Score</div>
          <div className="text-6xl font-black text-primary mb-4">{team.totalPoints}</div>
          <div className="inline-flex items-center gap-2 text-lg font-medium px-4 py-2 rounded-full bg-card border border-border">
            Rank: <span className="text-2xl">{getRankIcon(rank)} {rank}</span>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-20 md:pb-0">
      {/* Header Profile */}
      <div className="glass rounded-2xl p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3" />
        <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Badge variant="gold" className="text-xs px-2 py-0.5 shadow-lg">Current Rank: {getRankIcon(rank)} {rank}</Badge>
              <Badge variant="outline" className="font-mono text-xs">{team.teamId}</Badge>
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight">{team.name}</h1>
            <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
              <Users className="w-4 h-4" />
              <span>{team.members.length} Members</span>
            </div>
          </div>
          <div className="flex flex-col items-end">
            <div className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Total Points</div>
            <div className="text-5xl font-black gradient-text drop-shadow-sm">{team.totalPoints}</div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left Column: Current Objective & Progress */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Current Location / Objective */}
          <Card className="glass overflow-hidden relative border-primary/20">
            <div className="absolute top-0 left-0 w-1 h-full bg-primary" />
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5 text-primary" />
                  Current Objective
                </CardTitle>
                {route && (
                  <Badge variant="secondary">Step {route.currentStep + 1} of {route.totalSteps}</Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {currentLocation ? (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-2xl font-bold mb-2">{currentLocation.name}</h3>
                    <p className="text-muted-foreground leading-relaxed">{currentLocation.description}</p>
                  </div>

                  {currentLocation.clueUnlocked ? (
                    <div className="p-5 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
                      <div className="flex items-center gap-2 text-emerald-500 font-semibold mb-2">
                        <CheckCircle2 className="w-5 h-5" /> Activity Approved! Clue Unlocked:
                      </div>
                      <p className="font-medium italic leading-relaxed text-lg">"{currentLocation.clue}"</p>
                      <p className="text-sm text-muted-foreground mt-3 pt-3 border-t border-emerald-500/20">
                        Solve this clue to find your next location. Scan the QR code there to continue!
                      </p>
                    </div>
                  ) : (
                    <div className="p-5 rounded-xl bg-primary/5 border border-primary/20">
                      <div className="text-sm font-bold text-primary uppercase tracking-wider mb-2 flex items-center gap-2">
                        <Camera className="w-4 h-4" /> Fun Activity Challenge
                      </div>
                      <p className="font-medium leading-relaxed mb-4">{currentLocation.funActivity}</p>
                      
                      <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                        <div className="text-sm text-muted-foreground flex items-center gap-2">
                          <LockKeyhole className="w-4 h-4" /> Clue is locked
                        </div>
                        <Link href="/scan">
                          <Button variant="default" size="sm">Submit Proof</Button>
                        </Link>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Navigation className="w-12 h-12 mx-auto text-muted-foreground mb-4 opacity-50" />
                  <p className="text-lg font-medium">Head to your first location!</p>
                  <p className="text-muted-foreground text-sm mt-1">Check your route below and scan the QR code when you arrive.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Route Progress */}
          {route && (
            <Card className="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Map className="w-5 h-5" />
                  Route Progress
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-6">
                  <div className="flex justify-between text-sm mb-2 font-medium">
                    <span>{getProgressPercentage(route.currentStep, route.totalSteps)}% Completed</span>
                    <span>{route.currentStep} / {route.totalSteps} Locations</span>
                  </div>
                  <Progress value={getProgressPercentage(route.currentStep, route.totalSteps)} className="h-2" />
                </div>

                <div className="relative border-l-2 border-muted ml-3 space-y-6 pb-2">
                  {route.steps.map((step: any, idx: number) => {
                    const isCompleted = step.isCompleted
                    const isCurrent = step.isCurrent
                    const isFuture = !isCompleted && !isCurrent

                    return (
                      <div key={idx} className="relative pl-6">
                        {/* Timeline dot */}
                        <div className={`absolute -left-[9px] top-1 w-4 h-4 rounded-full border-2 bg-background flex items-center justify-center ${
                          isCompleted ? 'border-emerald-500' :
                          isCurrent ? 'border-primary' : 'border-muted'
                        }`}>
                          {isCompleted && <div className="w-2 h-2 rounded-full bg-emerald-500" />}
                          {isCurrent && <div className="w-2 h-2 rounded-full bg-primary pulse-dot" />}
                        </div>

                        <div className={`
                          p-3 rounded-lg border 
                          ${isCompleted ? 'bg-emerald-500/5 border-emerald-500/20' : 
                            isCurrent ? 'bg-primary/5 border-primary/30 shadow-[0_0_15px_rgba(var(--primary),0.1)]' : 
                            'bg-card border-border opacity-60'}
                        `}>
                          <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-1">Step {step.order + 1}</div>
                          <div className={`font-semibold ${isCompleted ? 'text-emerald-400' : isCurrent ? 'text-primary' : ''}`}>
                            {isCompleted || isCurrent ? step.locationName : '??? (Locked)'}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <Card className="glass">
            <CardHeader className="pb-3 border-b border-border/50">
              <CardTitle className="text-lg flex items-center justify-between">
                Notifications
                {data.notifications.length > 0 && (
                  <Badge variant="destructive" className="rounded-full w-5 h-5 p-0 flex items-center justify-center">{data.notifications.length}</Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border/50 max-h-[400px] overflow-y-auto">
                {data.notifications.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground text-sm">No new notifications</div>
                ) : (
                  data.notifications.map((n: any) => (
                    <div key={n.id} className="p-4 bg-primary/5">
                      <div className="font-medium text-sm mb-1">{n.title}</div>
                      <div className="text-xs text-muted-foreground">{n.message}</div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
