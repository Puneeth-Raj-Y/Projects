import { cookies } from 'next/headers'
import { successResponse } from '@/lib/api-response'

export async function POST() {
  const cookieStore = await cookies()
  cookieStore.delete('admin_token')
  cookieStore.delete('team_token')
  return successResponse({ message: 'Logged out successfully' })
}
