import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'
import QRCode from 'qrcode'
import { v4 as uuidv4 } from 'uuid'

// GET /api/admin/locations
export async function GET() {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const locations = await prisma.location.findMany({
    include: { qrCode: { include: { _count: { select: { scanLogs: true } } } } },
    orderBy: { orderIndex: 'asc' },
  })

  return successResponse(locations)
}

// POST /api/admin/locations
export async function POST(request: NextRequest) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  try {
    const { name, description, clue, funActivity, activityType, points, orderIndex } = await request.json()

    if (!name || !description || !clue || !funActivity) {
      return errorResponse('Name, description, clue, and fun activity are required')
    }

    const location = await prisma.$transaction(async (tx) => {
      const loc = await tx.location.create({
        data: {
          name,
          description,
          clue,
          funActivity,
          activityType: activityType || 'photo',
          points: points || 10,
          orderIndex: orderIndex || 0,
        },
      })

      // Auto-generate QR code for this location
      const code = uuidv4()
      const qrDataUrl = await QRCode.toDataURL(
        JSON.stringify({ locationId: loc.id, code }),
        { width: 500, margin: 4, color: { dark: '#000000', light: '#ffffff' } }
      )

      await tx.qRCode.create({
        data: { locationId: loc.id, code, imageUrl: qrDataUrl },
      })

      return loc
    })

    const full = await prisma.location.findUnique({
      where: { id: location.id },
      include: { qrCode: true },
    })

    return successResponse(full, 201)
  } catch (error) {
    console.error('[LOCATIONS_POST]', error)
    return errorResponse('Internal server error', 500)
  }
}
