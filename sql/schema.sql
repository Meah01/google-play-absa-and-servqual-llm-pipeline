-- ABSA Sentiment Pipeline Database Schema
-- This file is automatically executed when PostgreSQL container starts

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Applications table
CREATE TABLE IF NOT EXISTS apps (
    app_id VARCHAR(255) PRIMARY KEY,
    app_name VARCHAR(500) NOT NULL,
    category VARCHAR(100),
    developer VARCHAR(255),
    rating DECIMAL(2,1),
    installs VARCHAR(50),
    price VARCHAR(20),
    content_rating VARCHAR(50),
    last_updated DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    user_name VARCHAR(255),
    content TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    thumbs_up_count INTEGER DEFAULT 0,
    review_created_version VARCHAR(50),
    review_date DATE,
    reply_content TEXT,
    reply_date DATE,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    language VARCHAR(10) DEFAULT 'en',
    content_length INTEGER,
    is_spam BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Deep ABSA results table (from batch processing)
CREATE TABLE IF NOT EXISTS deep_absa (
    absa_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    aspect VARCHAR(100) NOT NULL,
    sentiment_score DECIMAL(4,3) CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    confidence_score DECIMAL(4,3) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    opinion_text TEXT,
    opinion_start_pos INTEGER,
    opinion_end_pos INTEGER,
    processing_model VARCHAR(100),
    processing_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Quick ABSA results table (from real-time processing)
CREATE TABLE IF NOT EXISTS quick_absa (
    quick_absa_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    aspects_sentiment JSONB NOT NULL,
    overall_sentiment DECIMAL(4,3),
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Daily aspect sentiment aggregations
CREATE TABLE IF NOT EXISTS daily_aspect_sentiment (
    aggregation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    aspect VARCHAR(100) NOT NULL,
    avg_sentiment DECIMAL(4,3),
    sentiment_count INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    trend_direction VARCHAR(20), -- 'improving', 'declining', 'stable'
    trend_magnitude DECIMAL(4,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_id, date, aspect)
);

-- Aspect definitions and configurations
CREATE TABLE IF NOT EXISTS aspects (
    aspect_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    aspect_name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),
    keywords TEXT[], -- Array of keywords for this aspect
    weight DECIMAL(3,2) DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing jobs tracking
CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- 'batch_absa', 'realtime_absa', 'data_scraping'
    app_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    records_processed INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics and monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,3),
    metric_unit VARCHAR(50),
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts and notifications
CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id VARCHAR(255) REFERENCES apps(app_id),
    alert_type VARCHAR(50) NOT NULL, -- 'sentiment_drop', 'high_volume', 'system_error'
    severity VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    message TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    is_resolved BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_reviews_app_id ON reviews(app_id);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_processed ON reviews(processed);
CREATE INDEX IF NOT EXISTS idx_deep_absa_review_id ON deep_absa(review_id);
CREATE INDEX IF NOT EXISTS idx_deep_absa_app_aspect ON deep_absa(app_id, aspect);
CREATE INDEX IF NOT EXISTS idx_quick_absa_review_id ON quick_absa(review_id);
CREATE INDEX IF NOT EXISTS idx_daily_sentiment_app_date ON daily_aspect_sentiment(app_id, date);
CREATE INDEX IF NOT EXISTS idx_daily_sentiment_aspect ON daily_aspect_sentiment(aspect);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_type ON processing_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON alerts(is_resolved) WHERE is_resolved = FALSE;

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_reviews_content_gin ON reviews USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_deep_absa_opinion_gin ON deep_absa USING gin(to_tsvector('english', opinion_text));

-- Insert default aspects
INSERT INTO aspects (aspect_name, category, keywords, weight, description) VALUES
('ui', 'interface', ARRAY['interface', 'design', 'layout', 'screen', 'ui', 'ux', 'menu', 'button', 'navigation'], 1.0, 'User interface and design aspects'),
('performance', 'functionality', ARRAY['speed', 'fast', 'slow', 'lag', 'freeze', 'crash', 'performance', 'responsive'], 1.2, 'App performance and speed'),
('battery', 'system', ARRAY['battery', 'drain', 'power', 'charging', 'energy', 'consumption'], 1.1, 'Battery usage and power consumption'),
('features', 'functionality', ARRAY['feature', 'function', 'capability', 'tool', 'option', 'functionality'], 1.0, 'App features and capabilities'),
('usability', 'experience', ARRAY['easy', 'difficult', 'intuitive', 'confusing', 'user-friendly', 'simple', 'complex'], 1.0, 'Ease of use and user experience'),
('stability', 'reliability', ARRAY['stable', 'unstable', 'reliable', 'bug', 'error', 'glitch', 'issue'], 1.3, 'App stability and reliability'),
('content', 'quality', ARRAY['content', 'quality', 'information', 'data', 'accuracy', 'relevant'], 1.0, 'Content quality and relevance'),
('support', 'service', ARRAY['support', 'help', 'customer', 'service', 'response', 'assistance'], 1.0, 'Customer support and service'),
('privacy', 'security', ARRAY['privacy', 'security', 'permission', 'data', 'safe', 'secure'], 1.2, 'Privacy and security concerns'),
('ads', 'monetization', ARRAY['ads', 'advertisement', 'popup', 'banner', 'commercial', 'marketing'], 0.8, 'Advertising and monetization')
ON CONFLICT (aspect_name) DO NOTHING;

-- Insert sample apps for testing
INSERT INTO apps (app_id, app_name, category, developer, rating) VALUES
('com.instagram.android', 'Instagram', 'Social', 'Meta Platforms, Inc.', 4.2),
('com.whatsapp', 'WhatsApp', 'Communication', 'WhatsApp LLC', 4.3),
('com.spotify.music', 'Spotify', 'Music & Audio', 'Spotify AB', 4.4),
('com.netflix.mediaclient', 'Netflix', 'Entertainment', 'Netflix, Inc.', 4.1),
('com.google.android.youtube', 'YouTube', 'Video Players & Editors', 'Google LLC', 4.0)
ON CONFLICT (app_id) DO NOTHING;

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_apps_updated_at BEFORE UPDATE ON apps
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_aspects_updated_at BEFORE UPDATE ON aspects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate content length
CREATE OR REPLACE FUNCTION set_content_length()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_length = LENGTH(NEW.content);
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically set content length
CREATE TRIGGER set_review_content_length BEFORE INSERT OR UPDATE ON reviews
    FOR EACH ROW EXECUTE FUNCTION set_content_length();

-- Create views for common queries
CREATE OR REPLACE VIEW review_summary AS
SELECT
    app_id,
    COUNT(*) as total_reviews,
    AVG(rating) as avg_rating,
    COUNT(*) FILTER (WHERE processed = TRUE) as processed_reviews,
    COUNT(*) FILTER (WHERE is_spam = TRUE) as spam_reviews,
    MAX(review_date) as latest_review_date
FROM reviews
GROUP BY app_id;

CREATE OR REPLACE VIEW daily_app_sentiment AS
SELECT
    app_id,
    date,
    AVG(avg_sentiment) as overall_sentiment,
    COUNT(DISTINCT aspect) as aspects_analyzed,
    SUM(sentiment_count) as total_reviews_analyzed
FROM daily_aspect_sentiment
GROUP BY app_id, date
ORDER BY app_id, date DESC;