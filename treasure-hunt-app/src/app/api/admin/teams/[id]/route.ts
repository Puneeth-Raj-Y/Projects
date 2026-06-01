import { NextRequest } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse, notFoundResponse } from '@/lib/api-response'

// GET /api/admin/teams/[id]
export async function GET(_: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  const team = await prisma.team.findUnique({
    where: { id },
    include: {
      members: true,
      teamRoute: {
        include: {
          route: { include: { steps: { include: { location: true }, orderBy: { stepOrder: 'asc' } } } },
        },
      },
      submissions: { include: { location: true }, orderBy: { createdAt: 'desc' } },
      pointsHistory: { orderBy: { createdAt: 'desc' } },
      notifications: { orderBy: { createdAt: 'desc' } },
    },
  })

  if (!team) return notFoundResponse('Team')
  return successResponse(team)
}

// PUT /api/admin/teams/[id]
export async function PUT(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  try {
    const { name, email, contactNumber, status, password, members } = await request.json()

    const updateData: Record<string, unknown> = {}
    if (name) updateData.name = name
    if (email) updateData.email = email
    if (contactNumber) updateData.contactNumber = contactNumber
    if (status) updateData.status = status
    if (password) updateData.password = await bcrypt.hash(password, 10)

    const team = await prisma.$transaction(async (tx) => {
      const updated = await tx.team.update({ where: { id }, data: updateData })

      if (members && Array.isArray(members)) {
        await tx.teamMember.deleteMany({ where: { teamId: id } })
        await tx.teamMember.createMany({
          data: members.map((m: { name: string; studentId: string; phone: string; email: string }) => ({
            teamId: id,
            name: m.name,
            studentId: m.studentId,
            phone: m.phone,
            email: m.email,
          })),
        })
      }

      return updated
    })

    return successResponse(team)
  } catch (error) {
    console.error('[TEAMS_PUT]', error)
    return errorResponse('Internal server error', 500)
  }
}

// DELETE /api/admin/teams/[id]
export async function DELETE(_: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const { id } = await params
  try {
    await prisma.team.delete({ where: { id } })
    return successResponse({ message: 'Team deleted successfully' })
  } catch {
    return notFoundResponse('Team')
  }
}
