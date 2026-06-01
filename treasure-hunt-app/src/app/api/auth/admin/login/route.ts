import { NextRequest } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { signAdminToken } from '@/lib/auth'
import { successResponse, errorResponse } from '@/lib/api-response'
import { cookies } from 'next/headers'

export async function POST(request: NextRequest) {
  try {
    const { username, password } = await request.json()

    if (!username || !password) {
      return errorResponse('Username and password are required')
    }

    const admin = await prisma.admin.findUnique({ where: { username } })
    if (!admin) return errorResponse('Invalid credentials', 401)

    const valid = await bcrypt.compare(password, admin.password)
    if (!valid) return errorResponse('Invalid credentials', 401)

    const token = await signAdminToken({
      id: admin.id,
      username: admin.username,
      email: admin.email,
    })

    const cookieStore = await cookies()
    cookieStore.set('admin_token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24, // 24 hours
      path: '/',
    })

    return successResponse({ admin: { id: admin.id, username: admin.username, name: admin.name, email: admin.email } })
  } catch (error) {
    console.error('[ADMIN_LOGIN]', error)
    return errorResponse('Internal server error', 500)
  }
}
