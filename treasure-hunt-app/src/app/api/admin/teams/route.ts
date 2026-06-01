import { NextRequest } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'

// GET /api/admin/teams
export async function GET() {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const teams = await prisma.team.findMany({
    include: {
      members: true,
      teamRoute: {
        include: {
          route: { include: { steps: { include: { location: true }, orderBy: { stepOrder: 'asc' } } } },
        },
      },
      _count: { select: { submissions: true } },
    },
    orderBy: { totalPoints: 'desc' },
  })

  return successResponse(teams)
}

// POST /api/admin/teams
export async function POST(request: NextRequest) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  try {
    const { name, teamId, password, email, contactNumber, members } = await request.json()

    if (!name || !teamId || !password || !email || !contactNumber) {
      return errorResponse('All fields are required')
    }

    if (members && (members.length < 3 || members.length > 4)) {
      return errorResponse('Teams must have 3-4 members')
    }

    const existing = await prisma.team.findUnique({ where: { teamId } })
    if (existing) return errorResponse('Team ID already exists')

    const hashedPassword = await bcrypt.hash(password, 10)

    const team = await prisma.team.create({
      data: {
        name,
        teamId,
        password: hashedPassword,
        email,
        contactNumber,
        members: members
          ? {
              create: members.map((m: { name: string; studentId: string; phone: string; email: string }) => ({
                name: m.name,
                studentId: m.studentId,
                phone: m.phone,
                email: m.email,
              })),
            }
          : undefined,
      },
      include: { members: true },
    })

    return successResponse(team, 201)
  } catch (error) {
    console.error('[TEAMS_POST]', error)
    return errorResponse('Internal server error', 500)
  }
}
