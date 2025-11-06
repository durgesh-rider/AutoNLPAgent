FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 80

# Serve the built application
CMD ["npm", "run", "serve", "--", "--port", "80", "--host", "0.0.0.0"]