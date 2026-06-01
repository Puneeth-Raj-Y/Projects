import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, unauthorizedResponse } from '@/lib/api-response'

// GET /api/admin/analytics
export async function GET() {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const [
    totalTeams,
    activeTeams,
    completedTeams,
    totalLocations,
    totalSubmissions,
    pendingSubmissions,
    approvedSubmissions,
    teamPoints,
  ] = await Promise.all([
    prisma.team.count(),
    prisma.team.count({ where: { status: 'active' } }),
    prisma.team.count({ where: { status: 'completed' } }),
    prisma.location.count(),
    prisma.activitySubmission.count(),
    prisma.activitySubmission.count({ where: { status: 'pending' } }),
    prisma.activitySubmission.count({ where: { status: 'approved' } }),
    prisma.team.findMany({
      select: { id: true, name: true, teamId: true, totalPoints: true, status: true },
      orderBy: { totalPoints: 'desc' },
      take: 10,
    }),
  ])

  const completionRate = totalTeams > 0 ? Math.round((completedTeams / totalTeams) * 100) : 0

  // Completion times for completed teams
  const completedRoutes = await prisma.teamRoute.findMany({
    where: { isCompleted: true, startedAt: { not: null }, completedAt: { not: null } },
    select: { startedAt: true, completedAt: true },
  })

  let avgCompletionMinutes = 0
  if (completedRoutes.length > 0) {
    const totalMs = completedRoutes.reduce((sum, r) => {
      return sum + (r.completedAt!.getTime() - r.startedAt!.getTime())
    }, 0)
    avgCompletionMinutes = Math.round(totalMs / completedRoutes.length / 60000)
  }

  // Points over time (last 7 days)
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  const pointsOverTime = await prisma.points.groupBy({
    by: ['createdAt'],
    where: { createdAt: { gte: sevenDaysAgo } },
    _sum: { points: true },
  })

  return successResponse({
    overview: {
      totalTeams,
      activeTeams,
      completedTeams,
      totalLocations,
      totalSubmissions,
      pendingSubmissions,
      approvedSubmissions,
      completionRate,
      avgCompletionMinutes,
    },
    topTeams: teamPoints,
    pointsOverTime,
  })
}
