import { PrismaClient } from '@prisma/client'
import bcrypt from 'bcryptjs'
import QRCode from 'qrcode'
import { v4 as uuidv4 } from 'uuid'

const prisma = new PrismaClient()

async function main() {
  console.log('🌱 Seeding database...')

  // Game Settings
  await prisma.gameSettings.upsert({
    where: { id: 'default' },
    update: {},
    create: {
      id: 'default',
      autoApproveActivities: false,
      qrScanPoints: 5,
      activityPoints: 15,
      cluePoints: 10,
      routeCompletionPoints: 50,
      incorrectPenalty: 5,
      skipPenalty: 10,
    },
  })

  // Admin
  const adminPassword = await bcrypt.hash('admin123', 10)
  await prisma.admin.upsert({
    where: { username: 'admin' },
    update: {},
    create: {
      username: 'admin',
      email: 'admin@treasurehunt.com',
      password: adminPassword,
      name: 'Event Administrator',
    },
  })
  console.log('✅ Admin created — username: admin | password: admin123')

  // Locations
  const locationData = [
    {
      name: 'The Library Labyrinth',
      description: 'Find the hidden shelf in the college library, 3rd floor east wing.',
      clue: 'Your next destination echoes with the sound of machines. Head to the place where knowledge meets engineering.',
      funActivity: 'Take a group selfie with all team members making a "shh" gesture in front of the oldest book you can find!',
      activityType: 'photo',
      points: 20,
      orderIndex: 0,
    },
    {
      name: 'The Engineering Hub',
      description: 'Located at the main mechanical engineering workshop entrance.',
      clue: 'From gears to greens — where students relax between classes. Find the place where trees whisper secrets.',
      funActivity: 'Record a 15-second video of your team pretending to fix a "broken machine" with dramatic sound effects!',
      activityType: 'video',
      points: 25,
      orderIndex: 1,
    },
    {
      name: 'The Secret Garden',
      description: 'The botanical garden behind the science block.',
      clue: 'Numbers rule the next spot. A place where calculations echo through the halls.',
      funActivity: 'Take a creative photo of your team hiding behind a tree or bush — only eyes visible!',
      activityType: 'photo',
      points: 20,
      orderIndex: 2,
    },
    {
      name: 'The Math Department',
      description: 'Ground floor of the mathematics building, near the notice board.',
      clue: 'Where food fuels champions. The heart of the campus beats with the aroma of lunch.',
      funActivity: 'Answer this riddle: I have cities, but no houses live there. I have mountains, but no trees. I have water, but no fish. What am I? Submit your answer!',
      activityType: 'text',
      points: 30,
      orderIndex: 3,
    },
    {
      name: 'The Campus Cafeteria',
      description: 'Main canteen, near the center fountain.',
      clue: 'Look for the place where flags fly high and history is etched in stone.',
      funActivity: 'Take a photo of your team doing the most creative food arrangement on a tray!',
      activityType: 'photo',
      points: 20,
      orderIndex: 4,
    },
    {
      name: 'The Administration Block',
      description: 'Front of the main administrative building, under the flagpole.',
      clue: 'Your journey concludes where it all began — return to the starting point to claim your glory!',
      funActivity: 'Record a victory video — your team\'s celebration dance! Make it epic!',
      activityType: 'video',
      points: 35,
      orderIndex: 5,
    },
  ]

  const locations = []
  for (const loc of locationData) {
    const location = await prisma.location.upsert({
      where: { id: loc.name.toLowerCase().replace(/\s+/g, '-') },
      update: {},
      create: {
        id: loc.name.toLowerCase().replace(/\s+/g, '-'),
        ...loc,
      },
    })

    const code = uuidv4()
    const qrDataUrl = await QRCode.toDataURL(
      JSON.stringify({ locationId: location.id, code }),
      { width: 400, margin: 2, color: { dark: '#1a1a2e', light: '#ffffff' } }
    )

    await prisma.qRCode.upsert({
      where: { locationId: location.id },
      update: {},
      create: { locationId: location.id, code, imageUrl: qrDataUrl },
    })

    locations.push(location)
    console.log(`✅ Location: ${location.name}`)
  }

  // Teams
  const teamData = [
    {
      teamId: 'TEAM001',
      name: 'Phoenix Rising',
      password: 'phoenix123',
      email: 'phoenix@hunt.com',
      contactNumber: '9876543210',
      members: [
        { name: 'Arjun Kumar', studentId: '1RV21CS001', phone: '9876543211', email: 'arjun@student.com' },
        { name: 'Priya Sharma', studentId: '1RV21CS002', phone: '9876543212', email: 'priya@student.com' },
        { name: 'Rohit Verma', studentId: '1RV21CS003', phone: '9876543213', email: 'rohit@student.com' },
        { name: 'Sneha Patel', studentId: '1RV21CS004', phone: '9876543214', email: 'sneha@student.com' },
      ],
    },
    {
      teamId: 'TEAM002',
      name: 'Thunder Hawks',
      password: 'thunder123',
      email: 'thunder@hunt.com',
      contactNumber: '9876543220',
      members: [
        { name: 'Vikram Singh', studentId: '1RV21CS011', phone: '9876543221', email: 'vikram@student.com' },
        { name: 'Ananya Roy', studentId: '1RV21CS012', phone: '9876543222', email: 'ananya@student.com' },
        { name: 'Karan Mehta', studentId: '1RV21CS013', phone: '9876543223', email: 'karan@student.com' },
      ],
    },
    {
      teamId: 'TEAM003',
      name: 'Shadow Wolves',
      password: 'shadow123',
      email: 'shadow@hunt.com',
      contactNumber: '9876543230',
      members: [
        { name: 'Riya Gupta', studentId: '1RV21CS021', phone: '9876543231', email: 'riya@student.com' },
        { name: 'Aditya Nair', studentId: '1RV21CS022', phone: '9876543232', email: 'aditya@student.com' },
        { name: 'Meera Joshi', studentId: '1RV21CS023', phone: '9876543233', email: 'meera@student.com' },
        { name: 'Dev Reddy', studentId: '1RV21CS024', phone: '9876543234', email: 'dev@student.com' },
      ],
    },
  ]

  for (const td of teamData) {
    const hashedPwd = await bcrypt.hash(td.password, 10)
    const team = await prisma.team.upsert({
      where: { teamId: td.teamId },
      update: {},
      create: {
        teamId: td.teamId,
        name: td.name,
        password: hashedPwd,
        email: td.email,
        contactNumber: td.contactNumber,
        members: { create: td.members },
      },
    })
    console.log(`✅ Team: ${td.name} | ID: ${td.teamId} | Password: ${td.password}`)
  }

  console.log('\n🎉 Seed complete!')
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
  console.log('🔐 Admin Login: admin / admin123')
  console.log('👥 Team Logins:')
  teamData.forEach((t) => console.log(`   ${t.name}: ${t.teamId} / ${t.password}`))
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect())
