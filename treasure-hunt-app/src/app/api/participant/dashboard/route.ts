import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getTeamFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'

// GET /api/participant/dashboard
export async function GET() {
  const teamAuth = await getTeamFromCookie()
  if (!teamAuth) return unauthorizedResponse()

  const team = await prisma.team.findUnique({
    where: { id: teamAuth.id },
    include: {
      members: true,
      teamRoute: {
        include: {
          route: {
            include: {
              steps: {
                include: { location: { include: { qrCode: true } } },
                orderBy: { stepOrder: 'asc' },
              },
            },
          },
        },
      },
      submissions: {
        include: { location: true },
        orderBy: { createdAt: 'desc' },
        take: 10,
      },
      notifications: {
        where: { isRead: false },
        orderBy: { createdAt: 'desc' },
        take: 5,
      },
      pointsHistory: { orderBy: { createdAt: 'desc' }, take: 10 },
    },
  })

  if (!team) return errorResponse('Team not found', 404)

  // Get rank
  const allTeams = await prisma.team.findMany({
    select: { id: true, totalPoints: true },
    orderBy: { totalPoints: 'desc' },
  })
  const rank = allTeams.findIndex((t) => t.id === team.id) + 1

  // Determine current location and clue unlock status
  const route = team.teamRoute
  let currentLocation = null
  let clueUnlocked = false

  if (route) {
    const currentStep = route.currentStep
    const steps = route.route.steps
    if (currentStep < steps.length) {
      const step = steps[currentStep]
      currentLocation = step.location

      // Check if activity for current location is approved
      const approvedSubmission = team.submissions.find(
        (s) => s.locationId === step.locationId && s.status === 'approved'
      )
      clueUnlocked = !!approvedSubmission
    }
  }

  return successResponse({
    team: {
      id: team.id,
      name: team.name,
      teamId: team.teamId,
      status: team.status,
      totalPoints: team.totalPoints,
      members: team.members,
    },
    rank,
    route: team.teamRoute
      ? {
          currentStep: team.teamRoute.currentStep,
          totalSteps: team.teamRoute.route.steps.length,
          isCompleted: team.teamRoute.isCompleted,
          steps: team.teamRoute.route.steps.map((s, idx) => ({
            order: s.stepOrder,
            locationName: s.location.name,
            locationId: s.locationId,
            isCompleted: idx < team.teamRoute!.currentStep,
            isCurrent: idx === team.teamRoute!.currentStep,
          })),
        }
      : null,
    currentLocation: currentLocation
      ? {
          id: currentLocation.id,
          name: currentLocation.name,
          description: currentLocation.description,
          funActivity: currentLocation.funActivity,
          activityType: currentLocation.activityType,
          points: currentLocation.points,
          clue: clueUnlocked ? currentLocation.clue : null,
          clueUnlocked,
        }
      : null,
    recentSubmissions: team.submissions,
    notifications: team.notifications,
    pointsHistory: team.pointsHistory,
  })
}
