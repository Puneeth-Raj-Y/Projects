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
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingTeam, setEditingTeam] = useState<any | null>(null)
  const [submitting, setSubmitting] = useState(false)
  
  const initialMember = { name: '', studentId: '', phone: '', email: '' }
  const [formData, setFormData] = useState({
    name: '',
    teamId: '',
    password: '',
    email: '',
    contactNumber: '',
    status: 'ACTIVE',
    members: [ { ...initialMember }, { ...initialMember }, { ...initialMember }, { ...initialMember } ]
  })
  
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

  function handleAddClick() {
    setEditingTeam(null)
    setFormData({
      name: '',
      teamId: '',
      password: '',
      email: '',
      contactNumber: '',
      status: 'ACTIVE',
      members: [ { ...initialMember }, { ...initialMember }, { ...initialMember }, { ...initialMember } ]
    })
    setIsModalOpen(true)
  }

  function handleEditClick(team: any) {
    setEditingTeam(team)
    const currentMembers = team.members || []
    const paddedMembers = [...currentMembers]
    while (paddedMembers.length < 4) {
      paddedMembers.push({ ...initialMember })
    }
    setFormData({
      name: team.name,
      teamId: team.teamId,
      password: '',
      email: team.email,
      contactNumber: team.contactNumber,
      status: team.status,
      members: paddedMembers.slice(0, 4)
    })
    setIsModalOpen(true)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    
    const validMembers = formData.members.filter(m => m.name.trim() && m.studentId.trim())
    if (validMembers.length < 3) {
      toast({ title: 'Validation Error', description: 'At least 3 valid members are required.', variant: 'destructive' })
      return
    }
    
    if (!editingTeam && !formData.password) {
      toast({ title: 'Validation Error', description: 'Password is required for new teams.', variant: 'destructive' })
      return
    }

    setSubmitting(true)
    try {
      const url = editingTeam ? `/api/admin/teams/${editingTeam.id}` : '/api/admin/teams'
      const method = editingTeam ? 'PUT' : 'POST'
      
      const payload: any = { ...formData, members: validMembers }
      if (editingTeam && !payload.password) {
        delete payload.password // Don't update password if left blank during edit
      }

      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const json = await res.json()
      
      if (json.success) {
        toast({ 
          title: editingTeam ? 'Team Updated' : 'Team Created', 
          variant: 'success' as any 
        })
        setIsModalOpen(false)
        fetchTeams()
      } else {
        toast({ title: 'Error', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Network Error', variant: 'destructive' })
    } finally {
      setSubmitting(false)
    }
  }

  async function deleteTeam(id: string) {
    if (!confirm('Are you sure you want to delete this team? This action cannot be undone.')) return
    try {
      const res = await fetch(`/api/admin/teams/${id}`, { method: 'DELETE' })
      const json = await res.json()
      if (json.success) {
        toast({ title: 'Team deleted', variant: 'success' as any })
        fetchTeams()
      } else {
        toast({ title: 'Error deleting', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Network Error', variant: 'destructive' })
    }
  }

  function updateMember(index: number, field: string, value: string) {
    const newMembers = [...formData.members]
    newMembers[index] = { ...newMembers[index], [field]: value }
    setFormData({ ...formData, members: newMembers })
  }

  return (
    <div className="flex flex-col gap-6 relative">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Users className="w-6 h-6 text-primary" />
            Team Management
          </h1>
          <p className="text-muted-foreground text-sm">Manage participants and their progress.</p>
        </div>
        <Button variant="gradient" onClick={handleAddClick}>
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
                    <Button variant="outline" size="sm" onClick={() => handleEditClick(team)}>
                      <Edit className="w-3 h-3 mr-1" /> Manage
                    </Button>
                    <Button variant="ghost" size="sm" className="text-destructive px-2" onClick={() => deleteTeam(team.id)}>
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Add/Edit Team Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <Card className="w-full max-w-2xl glass border-amber-500/20 shadow-2xl relative animate-in fade-in zoom-in-95 duration-150 max-h-[90vh] flex flex-col">
            <button 
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground z-10"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            <CardHeader className="shrink-0 border-b border-border/50 pb-4">
              <CardTitle className="text-xl font-bold">
                {editingTeam ? 'Edit Team' : 'Add New Team'}
              </CardTitle>
              <CardDescription>
                {editingTeam ? 'Modify team details and members.' : 'Register a new team for the hunt.'}
              </CardDescription>
            </CardHeader>
            
            <CardContent className="overflow-y-auto p-6 space-y-6">
              <form id="team-form" onSubmit={handleSubmit} className="space-y-6">
                
                {/* Basic Details */}
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-primary uppercase tracking-wider">Team Details</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Team Name</label>
                      <input
                        type="text"
                        required
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Team ID (Login Username)</label>
                      <input
                        type="text"
                        required
                        disabled={!!editingTeam}
                        value={formData.teamId}
                        onChange={(e) => setFormData({ ...formData, teamId: e.target.value })}
                        className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                        {editingTeam ? 'Password (leave blank to keep current)' : 'Password'}
                      </label>
                      <input
                        type="text"
                        required={!editingTeam}
                        value={formData.password}
                        onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                        className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                    {editingTeam && (
                      <div className="space-y-1">
                        <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Status</label>
                        <select
                          value={formData.status}
                          onChange={(e) => setFormData({ ...formData, status: e.target.value })}
                          className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                          <option value="ACTIVE">ACTIVE</option>
                          <option value="COMPLETED">COMPLETED</option>
                          <option value="DISQUALIFIED">DISQUALIFIED</option>
                        </select>
                      </div>
                    )}
                    <div className="space-y-1">
                      <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Contact Email</label>
                      <input
                        type="email"
                        required
                        value={formData.email}
                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                        className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Contact Number</label>
                      <input
                        type="text"
                        required
                        value={formData.contactNumber}
                        onChange={(e) => setFormData({ ...formData, contactNumber: e.target.value })}
                        className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                  </div>
                </div>

                {/* Team Members */}
                <div className="space-y-4 pt-4 border-t border-border/50">
                  <div className="flex justify-between items-end">
                    <h3 className="text-sm font-semibold text-primary uppercase tracking-wider">Team Members (Min 3, Max 4)</h3>
                  </div>
                  
                  <div className="space-y-3">
                    {formData.members.map((member, index) => (
                      <div key={index} className="grid grid-cols-4 gap-2 bg-muted/30 p-2 rounded-lg border border-border/50">
                        <input
                          placeholder={`Member ${index + 1} Name`}
                          value={member.name}
                          onChange={(e) => updateMember(index, 'name', e.target.value)}
                          className="w-full bg-background border border-border rounded text-xs p-2 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <input
                          placeholder="Student ID"
                          value={member.studentId}
                          onChange={(e) => updateMember(index, 'studentId', e.target.value)}
                          className="w-full bg-background border border-border rounded text-xs p-2 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <input
                          placeholder="Phone"
                          value={member.phone}
                          onChange={(e) => updateMember(index, 'phone', e.target.value)}
                          className="w-full bg-background border border-border rounded text-xs p-2 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <input
                          placeholder="Email"
                          value={member.email}
                          onChange={(e) => updateMember(index, 'email', e.target.value)}
                          className="w-full bg-background border border-border rounded text-xs p-2 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                      </div>
                    ))}
                  </div>
                </div>

              </form>
            </CardContent>
            
            <div className="shrink-0 p-4 border-t border-border/50 flex justify-end gap-3 bg-background/50 rounded-b-xl">
              <Button type="button" variant="outline" onClick={() => setIsModalOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" form="team-form" variant="gradient" disabled={submitting}>
                {submitting ? 'Saving...' : (editingTeam ? 'Save Changes' : 'Create Team')}
              </Button>
            </div>
          </Card>
        </div>
      )}

    </div>
  )
}
