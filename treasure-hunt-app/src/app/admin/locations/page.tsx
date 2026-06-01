'use client'

import { useEffect, useState } from 'react'
import { MapPin, Plus, Download, Edit, Trash2, QrCode } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import Image from 'next/image'

export default function AdminLocationsPage() {
  const [locations, setLocations] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingLocation, setEditingLocation] = useState<any | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    clue: '',
    funActivity: '',
    activityType: 'photo',
    points: 10
  })
  const { toast } = useToast()

  async function fetchLocations() {
    setLoading(true)
    try {
      const res = await fetch('/api/admin/locations')
      const json = await res.json()
      if (json.success) setLocations(json.data)
    } catch {
      toast({ title: 'Error fetching locations', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLocations()
  }, [])

  function handleAddClick() {
    setEditingLocation(null)
    setFormData({
      name: '',
      description: '',
      clue: '',
      funActivity: '',
      activityType: 'photo',
      points: 10
    })
    setIsModalOpen(true)
  }

  function handleEditClick(loc: any) {
    setEditingLocation(loc)
    setFormData({
      name: loc.name,
      description: loc.description,
      clue: loc.clue,
      funActivity: loc.funActivity,
      activityType: loc.activityType,
      points: loc.points
    })
    setIsModalOpen(true)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!formData.name || !formData.description || !formData.clue || !formData.funActivity) {
      toast({ title: 'Validation Error', description: 'Please fill in all fields.', variant: 'destructive' })
      return
    }

    setSubmitting(true)
    try {
      const url = editingLocation ? `/api/admin/locations/${editingLocation.id}` : '/api/admin/locations'
      const method = editingLocation ? 'PUT' : 'POST'
      
      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          points: Number(formData.points)
        })
      })
      const json = await res.json()
      if (json.success) {
        toast({ 
          title: editingLocation ? 'Location Updated' : 'Location Created', 
          description: editingLocation ? 'Successfully saved location changes.' : 'Location and QR code generated successfully!', 
          variant: 'success' as any 
        })
        setIsModalOpen(false)
        fetchLocations()
      } else {
        toast({ title: 'Error saving location', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Network Error', variant: 'destructive' })
    } finally {
      setSubmitting(false)
    }
  }

  async function regenerateQR(id: string) {
    try {
      const res = await fetch(`/api/admin/locations/${id}`, { method: 'PATCH' })
      const json = await res.json()
      if (json.success) {
        toast({ title: 'QR Code Regenerated', variant: 'success' as any })
        fetchLocations()
      }
    } catch {
      toast({ title: 'Error regenerating QR', variant: 'destructive' })
    }
  }

  async function deleteLocation(id: string) {
    if (!confirm('Are you sure you want to delete this location? This may break existing routes.')) return
    
    try {
      const res = await fetch(`/api/admin/locations/${id}`, { method: 'DELETE' })
      const json = await res.json()
      if (json.success) {
        toast({ title: 'Location deleted', variant: 'success' as any })
        fetchLocations()
      } else {
        toast({ title: 'Error deleting', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Network Error', variant: 'destructive' })
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <MapPin className="w-6 h-6 text-primary" />
            Locations & QR Codes
          </h1>
          <p className="text-muted-foreground text-sm">Manage hunt locations and download QR codes.</p>
        </div>
        <Button variant="gradient" onClick={handleAddClick}>
          <Plus className="w-4 h-4 mr-2" />
          Add Location
        </Button>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="glass"><CardContent className="h-64 shimmer rounded-xl p-6" /></Card>
          ))}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
          {locations.map((loc) => (
            <Card key={loc.id} className="glass flex flex-col h-full card-hover">
              <CardHeader className="pb-3 border-b border-border/50">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg">{loc.name}</CardTitle>
                    <Badge variant="outline" className="mt-2 text-[10px]">
                      {loc.activityType.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="font-bold text-primary text-xl">{loc.points} <span className="text-sm font-normal text-muted-foreground">pts</span></div>
                </div>
              </CardHeader>
              <CardContent className="flex-1 p-5 flex flex-col">
                <div className="space-y-4 flex-1">
                  <div>
                    <div className="text-xs font-semibold text-muted-foreground mb-1 uppercase tracking-wider">Description</div>
                    <div className="text-sm line-clamp-2">{loc.description}</div>
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-muted-foreground mb-1 uppercase tracking-wider">Fun Activity</div>
                    <div className="text-sm line-clamp-2 italic">{loc.funActivity}</div>
                  </div>
                  <div className="bg-primary/5 rounded-lg p-3 border border-primary/20">
                    <div className="text-xs font-semibold text-primary mb-1 uppercase tracking-wider">Unlockable Clue</div>
                    <div className="text-sm font-medium line-clamp-2">{loc.clue}</div>
                  </div>
                </div>

                {/* QR Section */}
                <div className="mt-6 pt-5 border-t border-border/50 flex gap-4">
                  {loc.qrCode?.imageUrl ? (
                    <div className="w-24 h-24 rounded-lg bg-white p-2 flex-shrink-0 relative border border-border">
                      <Image src={loc.qrCode.imageUrl} alt="QR" fill className="object-contain" />
                    </div>
                  ) : (
                    <div className="w-24 h-24 rounded-lg bg-muted flex flex-col items-center justify-center flex-shrink-0 text-muted-foreground">
                      <QrCode className="w-8 h-8 opacity-50" />
                    </div>
                  )}
                  
                  <div className="flex-1 flex flex-col justify-center gap-2">
                    <div className="text-xs text-muted-foreground">
                      Scanned: <span className="font-bold text-foreground">{loc.qrCode?._count?.scanLogs || 0}</span> times
                    </div>
                    <div className="flex gap-2">
                      <a href={loc.qrCode?.imageUrl} download={`QR-${loc.name}.png`} className="flex-1">
                        <Button variant="outline" size="sm" className="w-full h-8 text-xs">
                          <Download className="w-3 h-3 mr-1" /> Download
                        </Button>
                      </a>
                      <Button variant="ghost" size="sm" className="h-8 px-2" onClick={() => regenerateQR(loc.id)}>
                        <QrCode className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end gap-2 mt-4">
                  <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => handleEditClick(loc)}>
                    <Edit className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="sm" className="text-destructive" onClick={() => deleteLocation(loc.id)}>
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Add Location Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <Card className="w-full max-w-lg glass border-amber-500/20 shadow-2xl relative animate-in fade-in zoom-in-95 duration-150">
            <button 
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            
            <CardHeader>
              <CardTitle className="text-xl font-bold">
                {editingLocation ? 'Edit Location' : 'Add New Location'}
              </CardTitle>
              <CardDescription>
                {editingLocation ? 'Modify checkout points, tasks, and clue details.' : 'Create a new treasure hunt check-point and auto-generate its QR code.'}
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Location Name</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Science Library, Old Fountain"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Description & Clue Clues</label>
                  <textarea
                    required
                    rows={2}
                    placeholder="Where is this location? Give a physical description."
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Activity Type</label>
                    <select
                      value={formData.activityType}
                      onChange={(e) => setFormData({ ...formData, activityType: e.target.value })}
                      className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    >
                      <option value="photo">Photo Upload</option>
                      <option value="video">Video Upload</option>
                      <option value="text">Text / Answer Answer</option>
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Points Reward</label>
                    <input
                      type="number"
                      min={5}
                      max={100}
                      value={formData.points}
                      onChange={(e) => setFormData({ ...formData, points: Number(e.target.value) })}
                      className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Fun Challenge/Task for Teams</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Strike a funny pose, Solve this riddle: 3+3*3"
                    value={formData.funActivity}
                    onChange={(e) => setFormData({ ...formData, funActivity: e.target.value })}
                    className="w-full bg-background border border-border rounded-lg h-10 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-primary">Unlockable Clue (Revealed upon completing task)</label>
                  <textarea
                    required
                    rows={2}
                    placeholder="Clue leading to the NEXT location..."
                    value={formData.clue}
                    onChange={(e) => setFormData({ ...formData, clue: e.target.value })}
                    className="w-full bg-background border border-border rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>

                <div className="flex justify-end gap-3 pt-4 border-t border-border">
                  <Button type="button" variant="outline" onClick={() => setIsModalOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" variant="gradient" disabled={submitting}>
                    {submitting ? 'Saving...' : (editingLocation ? 'Save Changes' : 'Create Location')}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
