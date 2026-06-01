import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse, notFoundResponse } from '@/lib/api-response'
import QRCode from 'qrcode'
import { v4 as uuidv4 } from 'uuid'

// GET /api/admin/locations/[id]
export async function GET(_: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  const location = await prisma.location.findUnique({
    where: { id },
    include: { qrCode: { include: { scanLogs: { orderBy: { scannedAt: 'desc' }, take: 20 } } } },
  })
  if (!location) return notFoundResponse('Location')
  return successResponse(location)
}

// PUT /api/admin/locations/[id]
export async function PUT(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  try {
    const body = await request.json()
    const location = await prisma.location.update({ where: { id }, data: body })
    return successResponse(location)
  } catch (error) {
    console.error('[LOCATIONS_PUT]', error)
    return errorResponse('Internal server error', 500)
  }
}

// DELETE /api/admin/locations/[id]
export async function DELETE(_: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  try {
    await prisma.$transaction(async (tx) => {
      // 1. Delete associated Route Steps first to clear FK constraint
      await tx.routeStep.deleteMany({
        where: { locationId: id }
      })

      // 2. Delete associated Activity Submissions
      await tx.activitySubmission.deleteMany({
        where: { locationId: id }
      })

      // 3. Delete QRCode (associated ScanLogs will cascade delete because of QRCode onDelete: Cascade)
      await tx.qRCode.deleteMany({
        where: { locationId: id }
      })

      // 4. Finally delete the location itself
      await tx.location.delete({
        where: { id }
      })
    })

    return successResponse({ message: 'Location deleted successfully' })
  } catch (error) {
    console.error('[LOCATION_DELETE_ERROR]', error)
    return errorResponse('Failed to delete location due to database dependencies', 500)
  }
}

// PATCH /api/admin/locations/[id] — regenerate QR
export async function PATCH(_: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  try {
    const code = uuidv4()
    const qrDataUrl = await QRCode.toDataURL(
      JSON.stringify({ locationId: id, code }),
      { width: 500, margin: 4, color: { dark: '#000000', light: '#ffffff' } }
    )

    const qr = await prisma.qRCode.upsert({
      where: { locationId: id },
      update: { code, imageUrl: qrDataUrl },
      create: { locationId: id, code, imageUrl: qrDataUrl },
    })

    return successResponse(qr)
  } catch (error) {
    console.error('[QR_REGEN]', error)
    return errorResponse('Internal server error', 500)
  }
}
