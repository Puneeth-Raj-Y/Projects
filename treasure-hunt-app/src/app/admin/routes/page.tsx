'use client'

import { useEffect, useState } from 'react'
import { Route, Play, RefreshCw, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'

export default function AdminRoutesPage() {
  const [teams, setTeams] = useState<any[]>([])
  const [locations, setLocations] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const { toast } = useToast()

  async function fetchData() {
    setLoading(true)
    try {
      const [teamsRes, locsRes] = await Promise.all([
        fetch('/api/admin/teams'),
        fetch('/api/admin/locations')
      ])
      const teamsJson = await teamsRes.json()
      const locsJson = await locsRes.json()
      
      if (teamsJson.success) setTeams(teamsJson.data)
      if (locsJson.success) setLocations(locsJson.data)
    } catch {
      toast({ title: 'Error fetching data', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  async function generateRoutes() {
    const teamsWithoutRoutes = teams.filter(t => !t.teamRoute)
    if (teamsWithoutRoutes.length === 0) {
      toast({ title: 'All teams have routes', description: 'No missing routes to generate.', variant: 'success' as any })
      return
    }

    if (locations.length === 0) {
      toast({ title: 'No locations available', description: 'Create locations first.', variant: 'destructive' })
      return
    }

    setGenerating(true)
    try {
      const res = await fetch('/api/admin/routes', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          teamIds: teamsWithoutRoutes.map(t => t.id),
          locationIds: locations.map(l => l.id)
        })
      })
      const json = await res.json()
      if (json.success) {
        toast({ title: 'Routes Generated', description: 'Successfully assigned randomized routes.', variant: 'success' as any })
        fetchData()
      } else {
        toast({ title: 'Error', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Error generating routes', variant: 'destructive' })
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Route className="w-6 h-6 text-primary" />
            Route Management
          </h1>
          <p className="text-muted-foreground text-sm">Assign randomized location routes to all teams.</p>
        </div>
        <Button variant="gradient" onClick={generateRoutes} disabled={generating}>
          {generating ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
          Generate Missing Routes
        </Button>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="glass"><CardContent className="h-48 shimmer rounded-xl p-6" /></Card>
          ))}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {teams.map((team) => {
            const hasRoute = !!team.teamRoute?.route?.steps
            const steps = hasRoute ? [...team.teamRoute.route.steps].sort((a: any, b: any) => a.stepOrder - b.stepOrder) : []

            return (
              <Card key={team.id} className="glass flex flex-col h-full card-hover">
                <CardHeader className="pb-3 border-b border-border/50">
                  <CardTitle className="text-xl flex items-center justify-between">
                    {team.name}
                    {hasRoute ? (
                      <Badge variant="success" className="h-5 px-1.5"><CheckCircle2 className="w-3 h-3 mr-1" /> Assigned</Badge>
                    ) : (
                      <Badge variant="secondary" className="h-5 px-1.5">No Route</Badge>
                    )}
                  </CardTitle>
                  <CardDescription className="font-mono text-xs">{team.teamId}</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 p-5">
                  {hasRoute ? (
                    <div className="relative border-l-2 border-muted ml-3 space-y-4 pb-2">
                      {steps.map((step: any, idx: number) => {
                        const isCompleted = team.teamRoute.isCompleted || idx < team.teamRoute.currentStep
                        const isActive = !team.teamRoute.isCompleted && idx === team.teamRoute.currentStep

                        return (
                          <div key={step.id} className="relative pl-4">
                            <div className={`absolute -left-[9px] top-1 w-4 h-4 rounded-full border-2 bg-background flex items-center justify-center ${
                              isCompleted 
                                ? 'border-emerald-500 bg-emerald-500/10' 
                                : isActive 
                                  ? 'border-amber-500 bg-amber-500/10 animate-pulse' 
                                  : 'border-muted'
                            }`}>
                              {isCompleted && <div className="w-2 h-2 rounded-full bg-emerald-500" />}
                              {isActive && <div className="w-2 h-2 rounded-full bg-amber-500" />}
                            </div>
                            <div className="text-xs font-bold text-muted-foreground mb-0.5">Step {step.stepOrder + 1}</div>
                            <div className={`text-sm ${
                              isCompleted 
                                ? 'text-emerald-400 font-medium line-through opacity-70' 
                                : isActive 
                                  ? 'text-amber-400 font-bold' 
                                  : 'text-foreground/80'
                            }`}>
                              {step.location.name}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground p-6">
                      <Route className="w-8 h-8 opacity-20 mb-2" />
                      <p className="text-sm">No route assigned.</p>
                      <p className="text-xs mt-1">Click generate to assign a unique random route.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
