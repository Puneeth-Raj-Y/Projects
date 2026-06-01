'use client'

import { useEffect, useState } from 'react'
import { ClipboardCheck, Check, X, Clock, Image as ImageIcon, Video, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import Image from 'next/image'

export default function AdminActivitiesPage() {
  const [submissions, setSubmissions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  async function fetchSubmissions() {
    setLoading(true)
    try {
      const res = await fetch('/api/admin/activities')
      const json = await res.json()
      if (json.success) setSubmissions(json.data)
    } catch {
      toast({ title: 'Error fetching submissions', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSubmissions()
  }, [])

  async function handleReview(id: string, status: 'approved' | 'rejected') {
    try {
      const res = await fetch(`/api/admin/activities`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ submissionId: id, status }),
      })
      const json = await res.json()
      if (json.success) {
        toast({ title: `Submission ${status}`, variant: status === 'approved' ? 'success' : ('destructive' as any) })
        fetchSubmissions()
      } else {
        toast({ title: 'Error', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Error processing review', variant: 'destructive' })
    }
  }

  const getIcon = (type: string) => {
    switch (type) {
      case 'photo': return <ImageIcon className="w-4 h-4" />
      case 'video': return <Video className="w-4 h-4" />
      case 'text': return <FileText className="w-4 h-4" />
      default: return <ClipboardCheck className="w-4 h-4" />
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <ClipboardCheck className="w-6 h-6 text-primary" />
            Activity Submissions
          </h1>
          <p className="text-muted-foreground text-sm">Review team activities to unlock their next clues.</p>
        </div>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="glass"><CardContent className="h-64 shimmer rounded-xl p-6" /></Card>
          ))}
        </div>
      ) : submissions.length === 0 ? (
        <Card className="glass">
          <CardContent className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <CheckCircle2 className="w-16 h-16 text-emerald-500/50 mb-4" />
            <p className="text-xl font-medium">All caught up!</p>
            <p className="text-sm">No pending submissions to review.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
          {submissions.map((sub) => (
            <Card key={sub.id} className="glass flex flex-col h-full overflow-hidden border-amber-500/30 shadow-[0_0_15px_rgba(245,158,11,0.05)]">
              <CardHeader className="bg-amber-500/5 pb-4 border-b border-amber-500/20">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg">{sub.team.name}</CardTitle>
                    <CardDescription className="flex items-center gap-1 mt-1 text-xs">
                      <Clock className="w-3 h-3" />
                      {new Date(sub.createdAt).toLocaleTimeString()}
                    </CardDescription>
                  </div>
                  <Badge variant="warning" className="flex items-center gap-1 uppercase text-[10px]">
                    {getIcon(sub.submissionType)} {sub.submissionType}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="flex-1 p-0 flex flex-col">
                <div className="p-4 border-b border-border/50 bg-card/50">
                  <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-1">Location</div>
                  <div className="font-medium text-sm">{sub.location.name}</div>
                  
                  <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider mt-3 mb-1">Challenge</div>
                  <div className="text-sm line-clamp-2 italic">{sub.location.funActivity}</div>
                </div>

                <div className="flex-1 p-4 flex flex-col items-center justify-center min-h-[160px] bg-black/20">
                  {sub.submissionType === 'text' ? (
                    <div className="w-full h-full flex items-center justify-center p-6 text-center text-lg font-medium italic">
                      "{sub.content}"
                    </div>
                  ) : sub.submissionType === 'photo' ? (
                    <div className="relative w-full h-48 rounded-lg overflow-hidden border border-border">
                      <Image src={sub.content} alt="Submission" fill className="object-cover" />
                    </div>
                  ) : (
                    <div className="w-full text-center p-4">
                      <a href={sub.content} target="_blank" rel="noreferrer">
                        <Button variant="outline" className="w-full">
                          <Video className="w-4 h-4 mr-2" /> View Video
                        </Button>
                      </a>
                    </div>
                  )}
                </div>

                <div className="p-4 flex gap-3 border-t border-border/50">
                  <Button 
                    variant="outline" 
                    className="flex-1 border-destructive text-destructive hover:bg-destructive hover:text-destructive-foreground"
                    onClick={() => handleReview(sub.id, 'rejected')}
                  >
                    <X className="w-4 h-4 mr-1" /> Reject
                  </Button>
                  <Button 
                    variant="default" 
                    className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white"
                    onClick={() => handleReview(sub.id, 'approved')}
                  >
                    <Check className="w-4 h-4 mr-1" /> Approve
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

function CheckCircle2(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  )
}
