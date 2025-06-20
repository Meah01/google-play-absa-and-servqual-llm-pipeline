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

-- Clear existing app data and insert new ecommerce apps
DELETE FROM reviews WHERE app_id IN (
    'com.instagram.android', 'com.whatsapp', 'com.spotify.music',
    'com.netflix.mediaclient', 'com.google.android.youtube'
);
DELETE FROM apps WHERE app_id IN (
    'com.instagram.android', 'com.whatsapp', 'com.spotify.music',
    'com.netflix.mediaclient', 'com.google.android.youtube'
);

-- Insert new ecommerce apps
INSERT INTO apps (app_id, app_name, category, developer, rating) VALUES
('com.amazon.mShop.android.shopping', 'Amazon Shopping', 'Shopping', 'Amazon Mobile LLC', 4.3),
('com.einnovation.temu', 'Temu: Shop Like a Billionaire', 'Shopping', 'PDD Holdings Inc.', 4.6),
('com.zzkko', 'SHEIN-Shopping Online', 'Shopping', 'Roadget Business PTE. LTD.', 4.5),
('com.ebay.mobile', 'eBay online shopping & selling', 'Shopping', 'eBay Mobile', 4.3),
('com.etsy.android', 'Etsy: A Special Marketplace', 'Shopping', 'Etsy, Inc.', 4.7)
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

-- Add SERVQUAL dimension column to aspects table
ALTER TABLE aspects ADD COLUMN IF NOT EXISTS servqual_dimension VARCHAR(50);

-- SERVQUAL scores table for dimension-level aggregation
CREATE TABLE IF NOT EXISTS servqual_scores (
    score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_id VARCHAR(255) NOT NULL REFERENCES apps(app_id) ON DELETE CASCADE,
    dimension VARCHAR(50) NOT NULL,
    sentiment_score DECIMAL(4,3),
    quality_score INTEGER CHECK (quality_score >= 1 AND quality_score <= 5),
    review_count INTEGER DEFAULT 0,
    date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_id, dimension, date)
);

-- Clear existing aspects and insert ecommerce aspects with SERVQUAL mappings
DELETE FROM aspects;

-- Insert ecommerce-focused aspects with SERVQUAL dimension mappings
INSERT INTO aspects (aspect_name, category, keywords, weight, description, servqual_dimension) VALUES

-- RELIABILITY dimension (Platform consistency, order processing accuracy)
('product_quality', 'product', ARRAY['quality', 'cheap', 'flimsy', 'durable', 'material', 'defective', 'broken', 'fake', 'authentic', 'counterfeit', 'well made', 'poor quality'], 1.3, 'Product quality, materials, and authenticity', 'reliability'),

('product_description', 'product', ARRAY['description', 'accurate', 'misleading', 'photos', 'images', 'size', 'color', 'different', 'expected', 'as described', 'not as shown'], 1.2, 'Accuracy of product descriptions and images', 'reliability'),

('app_performance', 'technical', ARRAY['slow', 'fast', 'lag', 'crash', 'freeze', 'loading', 'responsive', 'performance', 'smooth', 'glitchy'], 1.1, 'App speed and technical performance', 'reliability'),

-- ASSURANCE dimension (Security, trust, professional competence)
('customer_service', 'service', ARRAY['customer service', 'support', 'help', 'response', 'staff', 'representative', 'chat', 'email', 'helpful', 'rude'], 1.2, 'Customer service quality and responsiveness', 'assurance'),

('payment_security', 'security', ARRAY['payment', 'secure', 'safe', 'fraud', 'scam', 'credit card', 'paypal', 'security', 'secure payment'], 1.2, 'Payment security and fraud protection', 'assurance'),

('seller_trust', 'trust', ARRAY['seller', 'vendor', 'trustworthy', 'reliable', 'scammer', 'legitimate', 'verified', 'trusted seller'], 1.1, 'Seller reliability and trustworthiness', 'assurance'),

-- TANGIBLES dimension (Physical appearance and interface design)
('search_navigation', 'experience', ARRAY['search', 'find', 'browse', 'navigation', 'filter', 'category', 'menu', 'interface', 'ui', 'easy to find', 'hard to navigate'], 1.1, 'Search functionality and app navigation', 'tangibles'),

('app_usability', 'technical', ARRAY['easy', 'user friendly', 'intuitive', 'confusing', 'complicated', 'design', 'layout', 'simple', 'hard to use'], 1.0, 'App usability and design', 'tangibles'),

('product_variety', 'product', ARRAY['selection', 'variety', 'options', 'catalog', 'inventory', 'stock', 'availability', 'out of stock', 'limited options'], 1.0, 'Product selection and inventory availability', 'tangibles'),

-- EMPATHY dimension (Personal attention and customer understanding)
('return_refund', 'service', ARRAY['return', 'refund', 'exchange', 'money back', 'policy', 'hassle', 'easy return', 'return process', 'refund policy'], 1.3, 'Return and refund process', 'empathy'),

-- RESPONSIVENESS dimension (Speed of service and communication)
('shipping_delivery', 'logistics', ARRAY['shipping', 'delivery', 'fast', 'slow', 'delayed', 'arrived', 'package', 'tracking', 'logistics', 'quick delivery', 'late delivery'], 1.3, 'Shipping speed and delivery experience', 'responsiveness'),

('shipping_cost', 'logistics', ARRAY['shipping cost', 'free shipping', 'expensive shipping', 'delivery fee', 'shipping price', 'shipping charges'], 1.1, 'Shipping costs and fees', 'responsiveness'),

('order_tracking', 'communication', ARRAY['track', 'tracking', 'status', 'update', 'progress', 'whereabouts', 'location', 'track order'], 1.1, 'Order tracking and status updates', 'responsiveness'),

-- ADDITIONAL ecommerce aspects
('pricing_value', 'financial', ARRAY['price', 'expensive', 'cheap', 'affordable', 'value', 'money', 'cost', 'worth', 'overpriced', 'good deal', 'value for money'], 1.2, 'Product pricing and value for money', 'assurance'),

('checkout_process', 'experience', ARRAY['checkout', 'payment', 'cart', 'order', 'purchase', 'buy', 'easy', 'difficult', 'smooth', 'complicated', 'simple checkout'], 1.2, 'Checkout and payment process', 'tangibles')

ON CONFLICT (aspect_name) DO NOTHING;

-- SERVQUAL performance indexes
CREATE INDEX IF NOT EXISTS idx_servqual_scores_app_date ON servqual_scores(app_id, date);
CREATE INDEX IF NOT EXISTS idx_servqual_scores_dimension ON servqual_scores(dimension);
CREATE INDEX IF NOT EXISTS idx_aspects_servqual_dimension ON aspects(servqual_dimension);

-- SERVQUAL analysis views
CREATE OR REPLACE VIEW servqual_dimension_summary AS
SELECT
    app_id,
    dimension,
    DATE_TRUNC('week', date) as week,
    AVG(quality_score) as avg_quality_score,
    AVG(sentiment_score) as avg_sentiment_score,
    SUM(review_count) as total_reviews,
    MAX(date) as latest_date
FROM servqual_scores
GROUP BY app_id, dimension, DATE_TRUNC('week', date)
ORDER BY app_id, dimension, week DESC;

CREATE OR REPLACE VIEW app_servqual_profile AS
SELECT
    app_id,
    MAX(CASE WHEN dimension = 'reliability' THEN quality_score END) as reliability_score,
    MAX(CASE WHEN dimension = 'assurance' THEN quality_score END) as assurance_score,
    MAX(CASE WHEN dimension = 'tangibles' THEN quality_score END) as tangibles_score,
    MAX(CASE WHEN dimension = 'empathy' THEN quality_score END) as empathy_score,
    MAX(CASE WHEN dimension = 'responsiveness' THEN quality_score END) as responsiveness_score,
    date
FROM servqual_scores
GROUP BY app_id, date
ORDER BY app_id, date DESC;

-- Add processing flags to reviews table for tracking completion status
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS absa_processed BOOLEAN DEFAULT FALSE;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS servqual_processed BOOLEAN DEFAULT FALSE;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS absa_processed_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS servqual_processed_at TIMESTAMP WITH TIME ZONE;

-- Processing checkpoints table for resume functionality
CREATE TABLE IF NOT EXISTS processing_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- 'absa', 'servqual', 'sequential'
    job_id UUID NOT NULL,
    reviews_processed INTEGER DEFAULT 0,
    total_reviews INTEGER DEFAULT 0,
    last_review_id UUID,
    current_app_id VARCHAR(255),
    checkpoint_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'paused', 'failed'
    batch_config JSONB, -- Store batch processing configuration
    progress_metadata JSONB, -- Additional progress tracking data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processing statistics table for detailed tracking
CREATE TABLE IF NOT EXISTS processing_statistics (
    stat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    app_id VARCHAR(255),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    reviews_processed INTEGER DEFAULT 0,
    aspects_extracted INTEGER DEFAULT 0,
    servqual_dimensions_updated INTEGER DEFAULT 0,
    failed_reviews INTEGER DEFAULT 0,
    processing_time_seconds DECIMAL(10,3),
    success_rate DECIMAL(5,2),
    error_details JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Progress notifications table for dashboard updates
CREATE TABLE IF NOT EXISTS progress_notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL,
    notification_type VARCHAR(50) NOT NULL, -- 'progress', 'complete', 'error', 'milestone'
    progress_percentage INTEGER,
    message TEXT NOT NULL,
    metadata JSONB,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Update the existing processing_jobs table with additional columns for sequential processing
DO $$
BEGIN
    -- Add columns if they don't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'sequential_mode') THEN
        ALTER TABLE processing_jobs ADD COLUMN sequential_mode BOOLEAN DEFAULT FALSE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'checkpoint_frequency') THEN
        ALTER TABLE processing_jobs ADD COLUMN checkpoint_frequency INTEGER DEFAULT 50;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'current_phase') THEN
        ALTER TABLE processing_jobs ADD COLUMN current_phase VARCHAR(20) DEFAULT 'pending'; -- 'pending', 'absa', 'servqual', 'completed'
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'phase_progress') THEN
        ALTER TABLE processing_jobs ADD COLUMN phase_progress JSONB;
    END IF;
END
$$;

-- Performance indexes for processing tables
CREATE INDEX IF NOT EXISTS idx_reviews_absa_processed ON reviews(absa_processed);
CREATE INDEX IF NOT EXISTS idx_reviews_servqual_processed ON reviews(servqual_processed);
CREATE INDEX IF NOT EXISTS idx_reviews_processing_status ON reviews(absa_processed, servqual_processed);
CREATE INDEX IF NOT EXISTS idx_reviews_app_processing ON reviews(app_id, absa_processed, servqual_processed);

CREATE INDEX IF NOT EXISTS idx_checkpoints_job_id ON processing_checkpoints(job_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON processing_checkpoints(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_type_status ON processing_checkpoints(job_type, status);

CREATE INDEX IF NOT EXISTS idx_processing_stats_job_id ON processing_statistics(job_id);
CREATE INDEX IF NOT EXISTS idx_processing_stats_app_type ON processing_statistics(app_id, job_type);
CREATE INDEX IF NOT EXISTS idx_processing_stats_time ON processing_statistics(start_time);

CREATE INDEX IF NOT EXISTS idx_notifications_job_id ON progress_notifications(job_id);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON progress_notifications(is_read) WHERE is_read = FALSE;

-- Views for processing monitoring
CREATE OR REPLACE VIEW processing_status_summary AS
SELECT
    pj.job_id,
    pj.job_type,
    pj.app_id,
    pj.status,
    pj.current_phase,
    pj.start_time,
    pj.end_time,
    pc.reviews_processed,
    pc.total_reviews,
    CASE
        WHEN pc.total_reviews > 0 THEN ROUND((pc.reviews_processed::DECIMAL / pc.total_reviews) * 100, 1)
        ELSE 0
    END as progress_percentage,
    ps.processing_time_seconds,
    ps.success_rate,
    pj.created_at
FROM processing_jobs pj
LEFT JOIN processing_checkpoints pc ON pj.job_id = pc.job_id AND pc.status = 'active'
LEFT JOIN processing_statistics ps ON pj.job_id = ps.job_id
ORDER BY pj.created_at DESC;

-- View for unprocessed reviews count by app
CREATE OR REPLACE VIEW unprocessed_reviews_summary AS
SELECT
    r.app_id,
    a.app_name,
    COUNT(*) FILTER (WHERE NOT r.absa_processed) as absa_pending,
    COUNT(*) FILTER (WHERE NOT r.servqual_processed) as servqual_pending,
    COUNT(*) FILTER (WHERE r.absa_processed AND NOT r.servqual_processed) as ready_for_servqual,
    COUNT(*) FILTER (WHERE r.absa_processed AND r.servqual_processed) as fully_processed,
    COUNT(*) as total_reviews,
    MIN(r.review_date) as oldest_unprocessed_date,
    MAX(r.review_date) as newest_unprocessed_date
FROM reviews r
INNER JOIN apps a ON r.app_id = a.app_id
WHERE NOT r.is_spam
GROUP BY r.app_id, a.app_name
ORDER BY absa_pending DESC, servqual_pending DESC;

-- Function to create processing checkpoint
CREATE OR REPLACE FUNCTION create_processing_checkpoint(
    p_job_id UUID,
    p_job_type VARCHAR(50),
    p_reviews_processed INTEGER,
    p_total_reviews INTEGER,
    p_last_review_id UUID DEFAULT NULL,
    p_current_app_id VARCHAR(255) DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    checkpoint_id UUID;
BEGIN
    -- Deactivate previous checkpoints for this job
    UPDATE processing_checkpoints
    SET status = 'completed'
    WHERE job_id = p_job_id AND status = 'active';

    -- Create new checkpoint
    INSERT INTO processing_checkpoints (
        job_id, job_type, reviews_processed, total_reviews,
        last_review_id, current_app_id, progress_metadata
    ) VALUES (
        p_job_id, p_job_type, p_reviews_processed, p_total_reviews,
        p_last_review_id, p_current_app_id, p_metadata
    ) RETURNING checkpoint_id INTO checkpoint_id;

    RETURN checkpoint_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get resumable processing state
CREATE OR REPLACE FUNCTION get_resumable_job_state(p_job_id UUID)
RETURNS TABLE (
    checkpoint_id UUID,
    reviews_processed INTEGER,
    total_reviews INTEGER,
    last_review_id UUID,
    current_app_id VARCHAR(255),
    progress_metadata JSONB,
    can_resume BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pc.checkpoint_id,
        pc.reviews_processed,
        pc.total_reviews,
        pc.last_review_id,
        pc.current_app_id,
        pc.progress_metadata,
        (pc.status = 'active' AND pj.status IN ('running', 'paused')) as can_resume
    FROM processing_checkpoints pc
    INNER JOIN processing_jobs pj ON pc.job_id = pj.job_id
    WHERE pc.job_id = p_job_id
    AND pc.status = 'active'
    ORDER BY pc.checkpoint_time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old checkpoints (run periodically)
CREATE OR REPLACE FUNCTION cleanup_old_checkpoints(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete old completed checkpoints
    DELETE FROM processing_checkpoints
    WHERE status IN ('completed', 'failed')
    AND checkpoint_time < CURRENT_TIMESTAMP - (days_to_keep || ' days')::interval;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Clean up orphaned notifications
    DELETE FROM progress_notifications
    WHERE created_at < CURRENT_TIMESTAMP - (days_to_keep || ' days')::interval;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add trigger to automatically update review processing timestamps
CREATE OR REPLACE FUNCTION update_processing_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.absa_processed = TRUE AND (OLD.absa_processed IS NULL OR OLD.absa_processed = FALSE) THEN
        NEW.absa_processed_at = CURRENT_TIMESTAMP;
    END IF;

    IF NEW.servqual_processed = TRUE AND (OLD.servqual_processed IS NULL OR OLD.servqual_processed = FALSE) THEN
        NEW.servqual_processed_at = CURRENT_TIMESTAMP;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_processing_timestamps
    BEFORE UPDATE ON reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_processing_timestamps();