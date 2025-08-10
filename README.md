# Snobbots Backend API

A Python FastAPI backend with Supabase authentication for Snobbots.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Supabase Authentication**: Secure user authentication with email/password
- **User Management**: Registration, login, and password reset functionality
- **Database Integration**: Automatic user registration in custom tables
- **Environment Configuration**: Flexible configuration using environment variables
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Logging**: Comprehensive logging for debugging and monitoring

## Project Structure

```
snobbots_backend/
├── app/
│   ├── __init__.py                    # Main app package
│   ├── main.py                        # FastAPI application entry point
│   │
│   ├── core/                          # Core application components
│   │   ├── __init__.py
│   │   └── config.py                  # Environment configuration & settings
│   │
│   ├── auth/                          # Authentication module
│   │   ├── __init__.py                # Auth package exports
│   │   ├── models.py                  # Auth-related Pydantic models
│   │   ├── auth_service.py            # Authentication business logic
│   │   └── auth_routes.py             # Authentication API endpoints
│   │
│   └── supabase/                      # Supabase integration
│       ├── __init__.py                # Supabase package exports
│       └── supabase_client.py         # Supabase client configuration
│
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
└── README.md                          # Project documentation
```

### 📁 **Folder Structure Explanation**

- **`app/`** - Main application package
  - **`core/`** - Core application components (config, utilities)
  - **`auth/`** - Authentication module (models, services, routes)
  - **`supabase/`** - Supabase integration and client setup
- **`requirements.txt`** - Python package dependencies
- **`.env.example`** - Environment variables template

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```bash
cp .env.example .env
```

Update the following variables in `.env`:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key
- `SECRET_KEY`: A random secret key for your application

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Supabase Database Setup

Create the following table in your Supabase database:

```sql
-- Create registered_users table
CREATE TABLE registered_users (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  approved BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE registered_users ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can read own profile" ON registered_users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON registered_users
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Allow insert for authenticated users" ON registered_users
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Add index for performance
CREATE INDEX idx_registered_users_email ON registered_users(email);
```

### 4. Run the Application

```bash
# Development mode (with auto-reload)
python app/main.py

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Authentication

All authentication endpoints are prefixed with `/api/auth`:

#### POST `/api/auth/register`

Register a new user.

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe"
}
```

**Response:**

```json
{
  "success": true,
  "message": "User registered successfully. You can now log in."
}
```

#### POST `/api/auth/login`

Login an existing user.

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Login successful"
}
```

#### POST `/api/auth/reset-password`

Send password reset email.

**Request Body:**

```json
{
  "email": "user@example.com"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Password reset email sent successfully"
}
```

#### GET `/api/auth/health`

Health check for auth service.

**Response:**

```json
{
  "status": "healthy",
  "service": "auth"
}
```

### General

#### GET `/`

Root endpoint with basic API information.

#### GET `/health`

General health check endpoint.

## API Documentation

Once the server is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Development

### Adding New Endpoints

1. Create new route files in the `app/` directory
2. Import and include them in `app/main.py`
3. Follow the existing pattern for error handling and logging

### Environment Variables

All configuration is managed through environment variables in `app/config.py`. Add new settings to the `Settings` class as needed.

### Error Handling

The application includes comprehensive error handling:

- Global exception handler for unexpected errors
- HTTP exceptions for API errors
- Detailed logging for debugging

## Production Deployment

For production deployment:

1. Set `DEBUG=False` in your environment
2. Use a proper ASGI server like Gunicorn with Uvicorn workers
3. Set up proper environment variable management
4. Configure logging for your environment
5. Set up monitoring and health checks
