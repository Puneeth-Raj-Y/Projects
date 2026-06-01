import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse, notFoundResponse } from '@/lib/api-response'

// GET /api/admin/activities — all submissions pending review
export async function GET(request: NextRequest) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { searchParams } = new URL(request.url)
  const status = searchParams.get('status') || 'pending'

  const submissions = await prisma.activitySubmission.findMany({
    where: status !== 'all' ? { status } : {},
    include: {
      team: { select: { id: true, name: true, teamId: true } },
      location: { select: { id: true, name: true, activityType: true, points: true } },
    },
    orderBy: { createdAt: 'desc' },
  })

  return successResponse(submissions)
}

// PATCH /api/admin/activities/[id]/review — approve/reject
export async function PATCH(request: NextRequest) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  try {
    const { submissionId, action, comment, bonusPoints, penaltyPoints } = await request.json()

    if (!submissionId || !action) {
      return errorResponse('Submission ID and action are required')
    }

    const validActions = ['approved', 'rejected', 'resubmit']
    if (!validActions.includes(action)) {
      return errorResponse('Invalid action. Use: approved, rejected, or resubmit')
    }

    const submission = await prisma.activitySubmission.findUnique({
      where: { id: submissionId },
      include: { location: true, team: true },
    })
    if (!submission) return notFoundResponse('Submission')

    const result = await prisma.$transaction(async (tx) => {
      // Update submission status
      const updated = await tx.activitySubmission.update({
        where: { id: submissionId },
        data: {
          status: action,
          adminComment: comment,
          bonusPoints: bonusPoints || 0,
          penaltyPoints: penaltyPoints || 0,
          reviewedAt: new Date(),
        },
      })

      if (action === 'approved') {
        const basePoints = submission.location.points
        const bonus = bonusPoints || 0
        const penalty = penaltyPoints || 0
        const totalPoints = basePoints + bonus - penalty

        // Award points
        await tx.points.create({
          data: {
            teamId: submission.teamId,
            points: totalPoints,
            reason: `Activity completed at ${submission.location.name}`,
            type: 'award',
            locationId: submission.locationId,
          },
        })

        // Update team total
        await tx.team.update({
          where: { id: submission.teamId },
          data: { totalPoints: { increment: totalPoints } },
        })

        // Unlock next step — advance currentStep in teamRoute
        const teamRoute = await tx.teamRoute.findUnique({
          where: { teamId: submission.teamId },
          include: {
            route: { include: { steps: { orderBy: { stepOrder: 'asc' } } } },
          },
        })

        if (teamRoute) {
          const currentLocationId = teamRoute.route.steps[teamRoute.currentStep]?.locationId
          if (currentLocationId === submission.locationId) {
            const nextStep = teamRoute.currentStep + 1
            const isCompleted = nextStep >= teamRoute.route.steps.length

            await tx.teamRoute.update({
              where: { teamId: submission.teamId },
              data: {
                currentStep: nextStep,
                isCompleted,
                completedAt: isCompleted ? new Date() : null,
                startedAt: teamRoute.startedAt || new Date(),
              },
            })

            if (isCompleted) {
              // Bonus for completing full route
              const settings = await tx.gameSettings.findFirst()
              const completionBonus = settings?.routeCompletionPoints || 50
              await tx.points.create({
                data: {
                  teamId: submission.teamId,
                  points: completionBonus,
                  reason: 'Route completion bonus!',
                  type: 'bonus',
                },
              })
              await tx.team.update({
                where: { id: submission.teamId },
                data: { totalPoints: { increment: completionBonus }, status: 'completed' },
              })
            }
          }
        }

        // Notify team
        await tx.notification.create({
          data: {
            teamId: submission.teamId,
            title: '✅ Activity Approved!',
            message: `Your submission at ${submission.location.name} was approved! +${totalPoints} points earned.${bonus > 0 ? ` Bonus: +${bonus}` : ''}`,
            type: 'success',
          },
        })
      } else if (action === 'rejected') {
        await tx.notification.create({
          data: {
            teamId: submission.teamId,
            title: '❌ Activity Rejected',
            message: `Your submission at ${submission.location.name} was rejected.${comment ? ` Reason: ${comment}` : ''}`,
            type: 'error',
          },
        })
      } else if (action === 'resubmit') {
        await tx.notification.create({
          data: {
            teamId: submission.teamId,
            title: '🔄 Resubmission Required',
            message: `Please resubmit your activity at ${submission.location.name}.${comment ? ` Note: ${comment}` : ''}`,
            type: 'warning',
          },
        })
      }

      return updated
    })

    return successResponse(result)
  } catch (error) {
    console.error('[ACTIVITIES_REVIEW]', error)
    return errorResponse('Internal server error', 500)
  }
}
