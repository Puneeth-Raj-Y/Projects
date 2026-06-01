import { NextRequest } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { signTeamToken } from '@/lib/auth'
import { successResponse, errorResponse } from '@/lib/api-response'
import { cookies } from 'next/headers'

export async function POST(request: NextRequest) {
  try {
    const { teamId, password } = await request.json()

    if (!teamId || !password) {
      return errorResponse('Team ID and password are required')
    }

    const team = await prisma.team.findUnique({
      where: { teamId },
      include: { members: true },
    })
    if (!team) return errorResponse('Invalid credentials', 401)

    const valid = await bcrypt.compare(password, team.password)
    if (!valid) return errorResponse('Invalid credentials', 401)

    if (team.status === 'disqualified') {
      return errorResponse('Your team has been disqualified', 403)
    }

    const token = await signTeamToken({
      id: team.id,
      teamId: team.teamId,
      name: team.name,
    })

    const cookieStore = await cookies()
    cookieStore.set('team_token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24,
      path: '/',
    })

    return successResponse({
      team: {
        id: team.id,
        teamId: team.teamId,
        name: team.name,
        status: team.status,
        totalPoints: team.totalPoints,
        members: team.members,
      },
    })
  } catch (error) {
    console.error('[TEAM_LOGIN]', error)
    return errorResponse('Internal server error', 500)
  }
}
