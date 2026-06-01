import Link from 'next/link'
import { MapPin, Users, Zap, Trophy, QrCode, Shield, ArrowRight, Star } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background animated-bg">
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-border/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center">
              <MapPin className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-lg gradient-text">TreasureQuest</span>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/login">
              <Button variant="ghost" size="sm">Team Login</Button>
            </Link>
            <Link href="/admin/login">
              <Button variant="gradient" size="sm">Admin Panel</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative pt-32 pb-20 px-4 text-center overflow-hidden">
        {/* Floating orbs */}
        <div className="absolute top-20 left-1/4 w-64 h-64 rounded-full bg-purple-600/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-10 right-1/4 w-80 h-80 rounded-full bg-amber-500/10 blur-3xl pointer-events-none" />

        <div className="relative max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium mb-6">
            <Zap className="w-3.5 h-3.5" />
            Gamified Event Management Platform
          </div>

          <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight mb-6 leading-tight">
            The Ultimate<br />
            <span className="gradient-text">Treasure Hunt</span><br />
            Platform
          </h1>

          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
            Run epic treasure hunts with QR code scanning, randomized routes,
            real-time leaderboards, and gamified activity challenges — all in one platform.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/admin/login">
              <Button variant="gradient" size="xl" className="w-full sm:w-auto">
                Start as Admin
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
            <Link href="/login">
              <Button variant="outline" size="xl" className="w-full sm:w-auto">
                Join as Team
                <Users className="w-5 h-5" />
              </Button>
            </Link>
          </div>
        </div>

        {/* Stats row */}
        <div className="mt-20 max-w-3xl mx-auto grid grid-cols-3 gap-6">
          {[
            { value: '100+', label: 'Events Run' },
            { value: '500+', label: 'Teams Played' },
            { value: '99%', label: 'Satisfaction' },
          ].map((stat) => (
            <div key={stat.label} className="glass rounded-xl p-4 card-hover">
              <div className="text-3xl font-extrabold gradient-text">{stat.value}</div>
              <div className="text-sm text-muted-foreground mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Everything You Need</h2>
            <p className="text-muted-foreground max-w-xl mx-auto">A complete platform built for real-world events.</p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: QrCode, title: 'QR Code System', desc: 'Auto-generate, scan, and validate unique QR codes for every location.', color: 'text-purple-400', bg: 'bg-purple-400/10' },
              { icon: MapPin, title: 'Smart Route Engine', desc: 'Randomized unique routes for every team to prevent copying.', color: 'text-blue-400', bg: 'bg-blue-400/10' },
              { icon: Trophy, title: 'Live Leaderboard', desc: 'Real-time rankings with points, progress, and completion times.', color: 'text-amber-400', bg: 'bg-amber-400/10' },
              { icon: Zap, title: 'Activity Challenges', desc: 'Photo, video, and text challenges with admin review workflow.', color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
              { icon: Shield, title: 'Clue Gating', desc: 'Clues only unlock after activity approval — no shortcuts!', color: 'text-red-400', bg: 'bg-red-400/10' },
              { icon: Users, title: 'Team Management', desc: 'Full CRUD for teams, members, scoring, and notifications.', color: 'text-indigo-400', bg: 'bg-indigo-400/10' },
            ].map((f) => (
              <div key={f.title} className="glass rounded-xl p-6 card-hover group">
                <div className={`w-12 h-12 rounded-xl ${f.bg} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <f.icon className={`w-6 h-6 ${f.color}`} />
                </div>
                <h3 className="font-semibold text-lg mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20 px-4 bg-card/30">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">How It Works</h2>
          <p className="text-muted-foreground mb-16">Simple for participants, powerful for admins.</p>
          <div className="grid sm:grid-cols-5 gap-4 items-center">
            {[
              { step: '1', label: 'Scan QR', icon: '📱' },
              { step: '→', label: '', icon: '' },
              { step: '2', label: 'Complete Activity', icon: '🎯' },
              { step: '→', label: '', icon: '' },
              { step: '3', label: 'Unlock Clue', icon: '🗝️' },
            ].map((item, i) => (
              item.step === '→' ? (
                <div key={i} className="text-2xl text-muted-foreground hidden sm:block">→</div>
              ) : (
                <div key={i} className="glass rounded-xl p-4 card-hover">
                  <div className="text-3xl mb-2">{item.icon}</div>
                  <div className="text-xs font-bold text-primary mb-1">Step {item.step}</div>
                  <div className="text-sm font-medium">{item.label}</div>
                </div>
              )
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 text-center">
        <div className="max-w-2xl mx-auto glass rounded-2xl p-12">
          <Star className="w-12 h-12 text-amber-400 mx-auto mb-4 float" />
          <h2 className="text-3xl font-bold mb-4">Ready to Run Your Hunt?</h2>
          <p className="text-muted-foreground mb-8">Login as admin to set up your event, or join as a team to start the adventure.</p>
          <div className="flex gap-4 justify-center">
            <Link href="/admin/login"><Button variant="gradient" size="lg">Admin Login</Button></Link>
            <Link href="/login"><Button variant="gold" size="lg">Team Login</Button></Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-8 text-center text-sm text-muted-foreground">
        <p>TreasureQuest — Built for epic events. © {new Date().getFullYear()}</p>
      </footer>
    </div>
  )
}
