# ── Build Stage ──
FROM node:20-alpine AS builder
WORKDIR /app

# Copy root configurations
COPY package*.json ./
COPY client/package*.json ./client/
COPY server/package*.json ./server/

# Install root dependencies
RUN npm install
RUN cd client && npm install
RUN cd server && npm install

# Copy application code
COPY . .

# Build Client production bundle
RUN npm run build

# ── Production Stage ──
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=5000

COPY --from=builder /app/package*.json ./
COPY --from=builder /app/server ./server

# Install only production dependencies
RUN cd server && npm install --only=production

EXPOSE 5000

CMD ["npm", "start"]
