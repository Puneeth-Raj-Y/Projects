/**
 * Random Route Allocation Engine
 * Generates unique randomized location sequences for each team.
 * Uses Fisher-Yates shuffle with collision avoidance to ensure
 * no two teams share identical routes.
 */

export function shuffleArray<T>(array: T[]): T[] {
  const arr = [...array]
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

/**
 * Generate a unique route that doesn't match any existing route.
 * Tries up to maxAttempts times before accepting a "least similar" result.
 */
export function generateUniqueRoute(
  locationIds: string[],
  existingRoutes: string[][], // each is an ordered array of location IDs
  maxAttempts: number = 50
): string[] {
  if (locationIds.length === 0) return []
  if (locationIds.length === 1) return [...locationIds]

  // The last location in the array is treated as the final destination common to all teams
  const finalLocationId = locationIds[locationIds.length - 1]
  const otherLocationIds = locationIds.slice(0, -1)

  let bestCandidate: string[] = []
  let bestScore = -1

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Shuffle all checkpoints except the final one
    const shuffledOthers = shuffleArray(otherLocationIds)
    // Append the common final destination to the end
    const candidate = [...shuffledOthers, finalLocationId]
    
    const score = minDifferenceScore(candidate, existingRoutes)

    if (score === candidate.length) {
      // Completely unique from all existing routes
      return candidate
    }

    if (score > bestScore) {
      bestScore = score
      bestCandidate = candidate
    }
  }

  return bestCandidate
}

/**
 * Returns the minimum number of positional differences
 * between the candidate and the closest existing route.
 */
function minDifferenceScore(candidate: string[], existingRoutes: string[][]): number {
  if (existingRoutes.length === 0) return candidate.length

  let minDiff = Infinity
  for (const existing of existingRoutes) {
    let matches = 0
    for (let i = 0; i < Math.min(candidate.length, existing.length); i++) {
      if (candidate[i] === existing[i]) matches++
    }
    const diff = candidate.length - matches
    if (diff < minDiff) minDiff = diff
  }
  return minDiff
}

/**
 * Assign routes to multiple teams at once, ensuring all teams get
 * as unique a route as possible.
 */
export function assignRoutesToTeams(
  teamIds: string[],
  locationIds: string[]
): Map<string, string[]> {
  const assignments = new Map<string, string[]>()
  const usedRoutes: string[][] = []

  for (const teamId of teamIds) {
    const route = generateUniqueRoute(locationIds, usedRoutes)
    assignments.set(teamId, route)
    usedRoutes.push(route)
  }

  return assignments
}
