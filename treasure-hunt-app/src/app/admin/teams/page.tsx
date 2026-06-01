'use client'

import { useEffect, useState } from 'react'
import { Users, Plus, Edit, Trash2, Mail, Phone, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import { getStatusColor, formatPoints } from '@/lib/utils'

export default function AdminTeamsPage() {
  const [teams, setTeams] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  async function fetchTeams() {
    setLoading(true)
    try {
      const res = await fetch('/api/admin/teams')
      const json = await res.json()
      if (json.success) setTeams(json.data)
    } catch {
      toast({ title: 'Error fetching teams', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTeams()
  }, [])

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Users className="w-6 h-6 text-primary" />
            Team Management
          </h1>
          <p className="text-muted-foreground text-sm">Manage participants and their progress.</p>
        </div>
        <Button variant="gradient">
          <Plus className="w-4 h-4 mr-2" />
          Add Team
        </Button>
      </div>

      {loading ? (
        <div className="grid lg:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="glass"><CardContent className="h-48 shimmer rounded-xl p-6" /></Card>
          ))}
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          {teams.map((team) => (
            <Card key={team.id} className="glass flex flex-col h-full card-hover">
              <CardHeader className="pb-3 border-b border-border/50">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-xl">{team.name}</CardTitle>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="font-mono text-xs text-muted-foreground">{team.teamId}</span>
                      <span className="w-1 h-1 rounded-full bg-border" />
                      <span className={getStatusColor(team.status)}>{team.status.toUpperCase()}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-primary text-2xl">{formatPoints(team.totalPoints)} <span className="text-sm font-normal text-muted-foreground">pts</span></div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 p-5 flex flex-col gap-4">
                <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1.5"><Mail className="w-4 h-4" /> {team.email}</div>
                  <div className="flex items-center gap-1.5"><Phone className="w-4 h-4" /> {team.contactNumber}</div>
                </div>

                <div className="bg-muted/50 rounded-lg p-3">
                  <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Members ({team.members.length})</div>
                  <div className="flex flex-wrap gap-2">
                    {team.members.map((m: any) => (
                      <Badge key={m.id} variant="secondary" className="text-xs font-normal">
                        {m.name}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between mt-auto pt-4">
                  <div className="text-xs text-muted-foreground">
                    Submissions: <span className="font-bold text-foreground">{team._count?.submissions || 0}</span>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">Manage <ChevronRight className="w-3 h-3 ml-1" /></Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
