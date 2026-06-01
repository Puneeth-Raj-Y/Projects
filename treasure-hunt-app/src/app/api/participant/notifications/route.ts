import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getTeamFromCookie } from '@/lib/auth'
import { successResponse, unauthorizedResponse } from '@/lib/api-response'

// GET /api/participant/notifications
export async function GET() {
  const teamAuth = await getTeamFromCookie()
  if (!teamAuth) return unauthorizedResponse()

  const notifications = await prisma.notification.findMany({
    where: { teamId: teamAuth.id },
    orderBy: { createdAt: 'desc' },
    take: 30,
  })

  return successResponse(notifications)
}

// PATCH /api/participant/notifications — mark all as read
export async function PATCH() {
  const teamAuth = await getTeamFromCookie()
  if (!teamAuth) return unauthorizedResponse()

  await prisma.notification.updateMany({
    where: { teamId: teamAuth.id, isRead: false },
    data: { isRead: true },
  })

  return successResponse({ message: 'All notifications marked as read' })
}
