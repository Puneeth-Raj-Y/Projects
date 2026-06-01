'use client'

import { useEffect, useState } from 'react'
import { Html5QrcodeScanner } from 'html5-qrcode'
import { Upload, Camera, FileText, CheckCircle2, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/hooks/use-toast'
import { useRouter } from 'next/navigation'

export default function ParticipantScanPage() {
  const [scanResult, setScanResult] = useState<any>(null)
  const [file, setFile] = useState<File | null>(null)
  const [textAnswer, setTextAnswer] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  useEffect(() => {
    // Initialize scanner only if we haven't scanned successfully yet
    if (scanResult) return

    const scanner = new Html5QrcodeScanner('reader', {
      qrbox: { width: 250, height: 250 },
      fps: 5,
    }, false)

    scanner.render(async (decodedText) => {
      scanner.clear()
      try {
        const res = await fetch('/api/participant/scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ qrData: decodedText }),
        })
        const json = await res.json()
        
        if (json.success) {
          toast({ title: 'Location Verified!', description: json.data.message, variant: 'success' as any })
          setScanResult(json.data)
        } else {
          toast({ title: 'Scan Failed', description: json.error, variant: 'destructive' })
          // Re-init scanner on failure after a short delay
          setTimeout(() => {
            window.location.reload()
          }, 2000)
        }
      } catch {
        toast({ title: 'Network Error', variant: 'destructive' })
      }
    }, (error) => {
      // Ignore routine scan errors (no QR found in frame)
    })

    return () => {
      scanner.clear().catch(console.error)
    }
  }, [scanResult, toast])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!scanResult) return
    
    const { location } = scanResult
    if (location.activityType !== 'text' && !file) {
      toast({ title: 'File required', description: 'Please upload your proof', variant: 'destructive' })
      return
    }
    if (location.activityType === 'text' && !textAnswer.trim()) {
      toast({ title: 'Answer required', description: 'Please enter your answer', variant: 'destructive' })
      return
    }

    setSubmitting(true)
    try {
      const formData = new FormData()
      formData.append('locationId', location.id)
      formData.append('submissionType', location.activityType)
      
      if (file) formData.append('file', file)
      if (textAnswer) formData.append('textAnswer', textAnswer)

      const res = await fetch('/api/participant/submit', {
        method: 'POST',
        body: formData,
      })
      const json = await res.json()

      if (json.success) {
        toast({ title: 'Success!', description: json.data.message, variant: 'success' as any })
        router.push('/dashboard')
      } else {
        toast({ title: 'Submission Failed', description: json.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Network Error', variant: 'destructive' })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto space-y-6 pb-20 md:pb-0">
      <div className="text-center space-y-2 mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight">Scanner</h1>
        <p className="text-muted-foreground">Scan the location QR code to unlock your activity challenge.</p>
      </div>

      {!scanResult ? (
        <Card className="glass overflow-hidden">
          <div className="scanner-overlay absolute inset-0 z-10 pointer-events-none" />
          <CardContent className="p-0">
            <div id="reader" className="w-full bg-black min-h-[300px] [&_video]:w-full [&_video]:object-cover" />
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6 animate-in slide-in-from-bottom-8 duration-500">
          <Card className="glass border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.1)]">
            <CardHeader className="bg-emerald-500/10 pb-4">
              <CardTitle className="flex items-center gap-2 text-emerald-500">
                <CheckCircle2 className="w-6 h-6" />
                Location Verified
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 space-y-4">
              <h2 className="text-2xl font-bold">{scanResult.location.name}</h2>
              <div className="p-4 rounded-xl bg-primary/5 border border-primary/20">
                <div className="text-xs font-bold text-primary uppercase tracking-wider mb-2">Challenge</div>
                <p className="font-medium text-lg leading-relaxed">{scanResult.location.funActivity}</p>
              </div>

              {scanResult.alreadyPending ? (
                <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-500 flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold mb-1">Submission Under Review</div>
                    <div className="text-sm opacity-90">You have already submitted proof for this location. Please wait for an admin to approve it to unlock your next clue.</div>
                  </div>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4 pt-4 border-t border-border">
                  {scanResult.location.activityType === 'text' ? (
                    <div className="space-y-2">
                      <Label htmlFor="answer">Your Answer</Label>
                      <Input
                        id="answer"
                        value={textAnswer}
                        onChange={e => setTextAnswer(e.target.value)}
                        placeholder="Type your answer here..."
                        className="h-12"
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Label>Upload Proof ({scanResult.location.activityType.toUpperCase()})</Label>
                      <div className="relative">
                        <input
                          type="file"
                          accept={scanResult.location.activityType === 'video' ? 'video/mp4,video/quicktime' : 'image/jpeg,image/png,image/webp'}
                          onChange={e => setFile(e.target.files?.[0] || null)}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                        />
                        <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl border-border bg-muted/30 group-hover:bg-muted/50 transition-colors">
                          <Upload className="w-8 h-8 text-muted-foreground mb-2" />
                          <div className="text-sm font-medium">
                            {file ? <span className="text-primary">{file.name}</span> : 'Tap to upload or take photo'}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <Button type="submit" variant="gradient" className="w-full h-12 text-base" disabled={submitting}>
                    {submitting ? 'Submitting...' : 'Submit to Unlock Clue'}
                  </Button>
                </form>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
