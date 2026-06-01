import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getTeamFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'
import { writeFile, mkdir } from 'fs/promises'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'

// POST /api/participant/submit — submit activity proof
export async function POST(request: NextRequest) {
  const teamAuth = await getTeamFromCookie()
  if (!teamAuth) return unauthorizedResponse()

  try {
    const formData = await request.formData()
    const locationId = formData.get('locationId') as string
    const submissionType = formData.get('submissionType') as string
    const textAnswer = formData.get('textAnswer') as string | null
    const file = formData.get('file') as File | null

    if (!locationId || !submissionType) {
      return errorResponse('Location ID and submission type are required')
    }

    // Check for existing pending submission
    const existing = await prisma.activitySubmission.findFirst({
      where: {
        teamId: teamAuth.id,
        locationId,
        status: { in: ['pending', 'approved'] },
      },
    })

    if (existing?.status === 'approved') {
      return errorResponse('Activity already approved for this location')
    }
    if (existing?.status === 'pending') {
      return errorResponse('You already have a pending submission for this location')
    }

    let fileUrl: string | null = null

    if (file && (submissionType === 'photo' || submissionType === 'video')) {
      const allowedTypes = {
        photo: ['image/jpeg', 'image/png', 'image/webp'],
        video: ['video/mp4', 'video/quicktime', 'video/mov'],
      }
      const allowed = allowedTypes[submissionType as 'photo' | 'video'] || []

      if (!allowed.includes(file.type)) {
        return errorResponse(`Invalid file type for ${submissionType}`)
      }

      const maxSize = submissionType === 'photo' ? 10 * 1024 * 1024 : 100 * 1024 * 1024
      if (file.size > maxSize) {
        return errorResponse(`File too large. Max: ${maxSize / 1024 / 1024}MB`)
      }

      const ext = file.name.split('.').pop()
      const filename = `${uuidv4()}.${ext}`
      const uploadDir = path.join(process.cwd(), 'public', 'uploads', teamAuth.id)

      await mkdir(uploadDir, { recursive: true })
      const buffer = Buffer.from(await file.arrayBuffer())
      await writeFile(path.join(uploadDir, filename), buffer)
      fileUrl = `/uploads/${teamAuth.id}/${filename}`
    }

    const settings = await prisma.gameSettings.findFirst()
    const autoApprove = settings?.autoApproveActivities || false

    const submission = await prisma.$transaction(async (tx) => {
      const sub = await tx.activitySubmission.create({
        data: {
          teamId: teamAuth.id,
          locationId,
          submissionType,
          fileUrl,
          textAnswer,
          status: autoApprove ? 'approved' : 'pending',
          reviewedAt: autoApprove ? new Date() : null,
        },
        include: { location: true },
      })

      if (autoApprove) {
        const pts = sub.location.points
        await tx.points.create({
          data: {
            teamId: teamAuth.id,
            points: pts,
            reason: `Auto-approved activity at ${sub.location.name}`,
            type: 'award',
            locationId,
          },
        })
        await tx.team.update({
          where: { id: teamAuth.id },
          data: { totalPoints: { increment: pts } },
        })

        // Advance route step
        const teamRoute = await tx.teamRoute.findUnique({
          where: { teamId: teamAuth.id },
          include: { route: { include: { steps: { orderBy: { stepOrder: 'asc' } } } } },
        })
        if (teamRoute) {
          const nextStep = teamRoute.currentStep + 1
          const isCompleted = nextStep >= teamRoute.route.steps.length
          await tx.teamRoute.update({
            where: { teamId: teamAuth.id },
            data: { currentStep: nextStep, isCompleted, completedAt: isCompleted ? new Date() : null },
          })
        }

        await tx.notification.create({
          data: {
            teamId: teamAuth.id,
            title: '✅ Activity Auto-Approved!',
            message: `Your submission at ${sub.location.name} was approved! +${pts} points. Clue unlocked!`,
            type: 'success',
          },
        })
      } else {
        await tx.notification.create({
          data: {
            teamId: teamAuth.id,
            title: '📤 Submission Received',
            message: `Your activity at ${sub.location.name} has been submitted for review.`,
            type: 'info',
          },
        })
      }

      return sub
    })

    return successResponse({
      submission: { id: submission.id, status: submission.status },
      autoApproved: autoApprove,
      message: autoApprove
        ? 'Activity approved! Clue unlocked!'
        : 'Submission received. Waiting for admin approval.',
    })
  } catch (error) {
    console.error('[SUBMIT]', error)
    return errorResponse('Internal server error', 500)
  }
}
