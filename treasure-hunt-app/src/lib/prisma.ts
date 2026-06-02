import { PrismaClient } from '@prisma/client'
import fs from 'fs'
import path from 'path'

// Auto-create directory for SQLite database if it doesn't exist
if (process.env.DATABASE_URL && process.env.DATABASE_URL.startsWith('file:')) {
  const cleanPath = process.env.DATABASE_URL.replace(/^file:/, '')
  if (cleanPath && (cleanPath.includes('/') || cleanPath.includes('\\'))) {
    const dirPath = path.dirname(cleanPath)
    if (!fs.existsSync(dirPath)) {
      try {
        fs.mkdirSync(dirPath, { recursive: true })
        console.log(`[Prisma] Created directory for database: ${dirPath}`)
      } catch (err) {
        console.error(`[Prisma] Failed to create directory: ${dirPath}`, err)
      }
    }
  }
}

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: ['query'],
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma
