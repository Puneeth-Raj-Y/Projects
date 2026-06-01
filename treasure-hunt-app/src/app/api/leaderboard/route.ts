import { prisma } from '@/lib/prisma'
import { successResponse } from '@/lib/api-response'

// GET /api/leaderboard — public leaderboard
export async function GET() {
  const teams = await prisma.team.findMany({
    select: {
      id: true,
      name: true,
      teamId: true,
      totalPoints: true,
      status: true,
      teamRoute: {
        select: {
          currentStep: true,
          isCompleted: true,
          startedAt: true,
          completedAt: true,
          route: {
            select: {
              _count: { select: { steps: true } },
            },
          },
        },
      },
      _count: {
        select: { submissions: true },
      },
    },
    orderBy: [{ totalPoints: 'desc' }, { updatedAt: 'asc' }],
  })

  const leaderboard = teams.map((team, index) => ({
    rank: index + 1,
    id: team.id,
    name: team.name,
    teamId: team.teamId,
    totalPoints: team.totalPoints,
    status: team.status,
    locationsCompleted: team.teamRoute?.currentStep ?? 0,
    totalLocations: team.teamRoute?.route?._count?.steps ?? 0,
    activitiesCompleted: team._count.submissions,
    isCompleted: team.teamRoute?.isCompleted ?? false,
    completionTime: team.teamRoute?.completedAt && team.teamRoute?.startedAt
      ? Math.round(
          (team.teamRoute.completedAt.getTime() - team.teamRoute.startedAt.getTime()) / 60000
        )
      : null,
  }))

  return successResponse(leaderboard)
}
