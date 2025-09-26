-- Test script for the Infera extension
-- This tests the hello_infera function

.echo on

-- Test 1: Basic function call
SELECT hello_infera() as greeting;

-- Test 2: Verify it returns a string
SELECT typeof(hello_infera()) as return_type;

-- Test 3: Use in a more complex query
SELECT
    'Test ID: ' || id as test_name,
    hello_infera() as rust_greeting
FROM (VALUES (1), (2), (3)) t(id);

-- Test 4: Combine with other functions
SELECT
    upper(hello_infera()) as uppercase_greeting,
    length(hello_infera()) as greeting_length;

.echo off
