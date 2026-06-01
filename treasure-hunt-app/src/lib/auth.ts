import { SignJWT, jwtVerify } from 'jose'
import { cookies } from 'next/headers'

const ADMIN_SECRET = new TextEncoder().encode(
  process.env.JWT_ADMIN_SECRET || 'treasure-hunt-admin-secret-2024'
)
const TEAM_SECRET = new TextEncoder().encode(
  process.env.JWT_TEAM_SECRET || 'treasure-hunt-team-secret-2024'
)

export interface AdminPayload {
  id: string
  username: string
  email: string
  role: 'admin'
}

export interface TeamPayload {
  id: string
  teamId: string
  name: string
  role: 'participant'
}

export async function signAdminToken(payload: Omit<AdminPayload, 'role'>) {
  return await new SignJWT({ ...payload, role: 'admin' })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('24h')
    .sign(ADMIN_SECRET)
}

export async function signTeamToken(payload: Omit<TeamPayload, 'role'>) {
  return await new SignJWT({ ...payload, role: 'participant' })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('24h')
    .sign(TEAM_SECRET)
}

export async function verifyAdminToken(token: string): Promise<AdminPayload | null> {
  try {
    const { payload } = await jwtVerify(token, ADMIN_SECRET)
    return payload as unknown as AdminPayload
  } catch {
    return null
  }
}

export async function verifyTeamToken(token: string): Promise<TeamPayload | null> {
  try {
    const { payload } = await jwtVerify(token, TEAM_SECRET)
    return payload as unknown as TeamPayload
  } catch {
    return null
  }
}

export async function getAdminFromCookie(): Promise<AdminPayload | null> {
  const cookieStore = await cookies()
  const token = cookieStore.get('admin_token')?.value
  if (!token) return null
  return verifyAdminToken(token)
}

export async function getTeamFromCookie(): Promise<TeamPayload | null> {
  const cookieStore = await cookies()
  const token = cookieStore.get('team_token')?.value
  if (!token) return null
  return verifyTeamToken(token)
}
