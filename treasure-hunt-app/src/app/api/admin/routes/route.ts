import { NextRequest } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getAdminFromCookie } from '@/lib/auth'
import { successResponse, errorResponse, unauthorizedResponse } from '@/lib/api-response'
import { generateUniqueRoute } from '@/lib/route-engine'
import { v4 as uuidv4 } from 'uuid'

// GET /api/admin/routes — list all routes
export async function GET() {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  const routes = await prisma.route.findMany({
    include: {
      steps: { include: { location: true }, orderBy: { stepOrder: 'asc' } },
      teamRoutes: { include: { team: true } },
    },
    orderBy: { createdAt: 'desc' },
  })

  return successResponse(routes)
}

// POST /api/admin/routes/generate — generate & assign random routes to all teams
export async function POST(request: NextRequest) {
  const admin = await getAdminFromCookie()
  if (!admin) return unauthorizedResponse()

  try {
    const { teamIds, locationIds } = await request.json()

    if (!teamIds?.length || !locationIds?.length) {
      return errorResponse('Team IDs and Location IDs are required')
    }

    const existingRoutes = await prisma.route.findMany({
      include: { steps: { orderBy: { stepOrder: 'asc' } } },
    })

    const usedSequences = existingRoutes.map((r) => r.steps.map((s) => s.locationId))

    const results = await prisma.$transaction(async (tx) => {
      const created = []

      for (const teamId of teamIds) {
        const sequence = generateUniqueRoute(locationIds, usedSequences)
        usedSequences.push(sequence)

        const route = await tx.route.create({
          data: {
            name: `Route-${uuidv4().slice(0, 8)}`,
            steps: {
              create: sequence.map((locId, idx) => ({
                locationId: locId,
                stepOrder: idx,
              })),
            },
          },
        })

        // Upsert team route assignment
        await tx.teamRoute.upsert({
          where: { teamId },
          update: { routeId: route.id, currentStep: 0, isCompleted: false, startedAt: null, completedAt: null },
          create: { teamId, routeId: route.id },
        })

        // Notify team
        await tx.notification.create({
          data: {
            teamId,
            title: 'Route Assigned!',
            message: 'Your treasure hunt route has been assigned. Get ready to start!',
            type: 'success',
          },
        })

        created.push({ teamId, routeId: route.id, steps: sequence })
      }

      return created
    })

    return successResponse(results, 201)
  } catch (error) {
    console.error('[ROUTES_GENERATE]', error)
    return errorResponse('Internal server error', 500)
  }
}
