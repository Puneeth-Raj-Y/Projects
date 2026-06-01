import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getTeamFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'

// POST /api/participant/scan — verify QR code scan
export async function POST(request: NextRequest) {
  const teamAuth = await getTeamFromCookie()
  if (!teamAuth) return unauthorizedResponse()

  try {
    const { qrData } = await request.json()

    if (!qrData) return errorResponse('QR data is required')

    let parsed: { locationId: string; code: string }
    try {
      parsed = typeof qrData === 'string' ? JSON.parse(qrData) : qrData
    } catch {
      return errorResponse('Invalid QR code format')
    }

    const { locationId, code } = parsed

    // Verify QR code
    const qrCode = await prisma.qRCode.findFirst({
      where: { locationId, code },
      include: { location: true },
    })

    if (!qrCode) return errorResponse('Invalid QR code', 400)

    // Get team route
    const teamRoute = await prisma.teamRoute.findUnique({
      where: { teamId: teamAuth.id },
      include: {
        route: { include: { steps: { orderBy: { stepOrder: 'asc' } } } },
      },
    })

    if (!teamRoute) return errorResponse('No route assigned to your team yet', 400)
    if (teamRoute.isLocked) return errorResponse('Your route is locked by admin', 400)
    if (teamRoute.isCompleted) return errorResponse('Your team has already completed the hunt!', 400)

    // Check if this is the correct next location
    const currentStep = teamRoute.route.steps[teamRoute.currentStep]
    if (!currentStep) return errorResponse('No more locations in your route', 400)

    if (currentStep.locationId !== locationId) {
      return errorResponse(
        `This is not your next location. Please follow your assigned route.`,
        400
      )
    }

    // Check if already scanned this step
    const existingSubmission = await prisma.activitySubmission.findFirst({
      where: {
        teamId: teamAuth.id,
        locationId,
        status: { in: ['pending', 'approved'] },
      },
    })

    if (existingSubmission?.status === 'approved') {
      return errorResponse('You have already completed this location', 400)
    }

    // Log the scan
    await prisma.$transaction(async (tx) => {
      await tx.qRCode.update({
        where: { id: qrCode.id },
        data: { scanCount: { increment: 1 } },
      })
      await tx.qRScanLog.create({
        data: { qrCodeId: qrCode.id, teamId: teamAuth.id },
      })

      // Award QR scan points
      const settings = await tx.gameSettings.findFirst()
      const scanPoints = settings?.qrScanPoints || 5

      await tx.points.create({
        data: {
          teamId: teamAuth.id,
          points: scanPoints,
          reason: `QR Scan at ${qrCode.location.name}`,
          type: 'award',
          locationId,
        },
      })
      await tx.team.update({
        where: { id: teamAuth.id },
        data: { totalPoints: { increment: scanPoints } },
      })

      // Start route timer if first scan
      if (!teamRoute.startedAt) {
        await tx.teamRoute.update({
          where: { teamId: teamAuth.id },
          data: { startedAt: new Date() },
        })
      }
    })

    return successResponse({
      location: {
        id: qrCode.location.id,
        name: qrCode.location.name,
        description: qrCode.location.description,
        funActivity: qrCode.location.funActivity,
        activityType: qrCode.location.activityType,
        points: qrCode.location.points,
      },
      message: 'Location verified! Complete the activity to unlock the clue.',
      alreadyPending: !!existingSubmission,
    })
  } catch (error) {
    console.error('[SCAN]', error)
    return errorResponse('Internal server error', 500)
  }
}
