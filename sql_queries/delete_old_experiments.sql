-- cleanup_queries.sql
-- This script contains queries for cleaning up MLflow database by deleting obsolete experiments and related runs.

-- Step 1: Delete metrics, parameters, and tags associated with deleted experiments
DELETE FROM metrics
WHERE run_uuid IN (
    SELECT run_uuid FROM runs
    WHERE experiment_id IN (SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted')
);

DELETE FROM params
WHERE run_uuid IN (
    SELECT run_uuid FROM runs
    WHERE experiment_id IN (SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted')
);

DELETE FROM tags
WHERE run_uuid IN (
    SELECT run_uuid FROM runs
    WHERE experiment_id IN (SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted')
);

-- Step 2: Delete runs associated with deleted experiments
DELETE FROM runs
WHERE experiment_id IN (
    SELECT experiment_id FROM experiments WHERE lifecycle_stage = 'deleted'
);

-- Step 3: Delete deleted experiments themselves
DELETE FROM experiments WHERE lifecycle_stage = 'deleted';
