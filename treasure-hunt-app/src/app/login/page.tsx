'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { MapPin, Lock, Trophy, Shield, Key } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useToast } from '@/hooks/use-toast'

export default function TeamLoginPage() {
  const [teamId, setTeamId] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const { toast } = useToast()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await fetch('/api/auth/team/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teamId, password }),
      })
      const data = await res.json()
      if (data.success) {
        toast({ title: `Welcome back, ${data.data.team.name}!`, variant: 'success' as any })
        router.push('/dashboard')
      } else {
        toast({ title: '❌ Login failed', description: data.error, variant: 'destructive' })
      }
    } catch {
      toast({ title: 'Error', description: 'Network error', variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background animated-bg flex items-center justify-center p-4">
      {/* Background orbs */}
      <div className="fixed top-1/4 left-1/4 w-96 h-96 rounded-full bg-amber-500/10 blur-3xl pointer-events-none" />
      <div className="fixed bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-orange-500/10 blur-3xl pointer-events-none" />

      <div className="w-full max-w-md relative">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 glow-gold mb-6 relative">
            <Trophy className="w-10 h-10 text-white" />
            <div className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full bg-background flex items-center justify-center">
              <MapPin className="w-4 h-4 text-primary" />
            </div>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Team Login</h1>
          <p className="text-muted-foreground mt-2">Enter your Team ID to continue your quest.</p>
        </div>

        <Card className="glass border-border/50 shadow-xl shadow-amber-500/5">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 text-sm font-medium text-amber-500">
              <Shield className="w-4 h-4" />
              <span>Participant Access</span>
            </div>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="teamId">Team ID</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="teamId"
                    placeholder="e.g. TEAM001"
                    value={teamId}
                    onChange={e => setTeamId(e.target.value.toUpperCase())}
                    className="pl-9 font-mono uppercase"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="pl-9"
                    required
                  />
                </div>
              </div>

              <Button type="submit" variant="gold" className="w-full text-base h-12" disabled={loading}>
                {loading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Connecting...
                  </div>
                ) : (
                  'Start Adventure'
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="mt-8 text-center text-sm text-muted-foreground">
          <p>Don't have a team ID? Contact your event organizer.</p>
          <Link href="/" className="inline-flex items-center gap-1 mt-2 text-primary hover:underline">
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  )
}
