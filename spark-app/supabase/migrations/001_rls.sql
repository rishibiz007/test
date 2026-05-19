-- Enable Row Level Security on all three tables.
-- The service_role key used in API routes bypasses RLS automatically.
-- These policies protect against data exposure if the anon key is ever
-- accidentally used client-side or leaked.

alter table lookup_cache  enable row level security;
alter table user_lookups  enable row level security;
alter table user_feedback enable row level security;

-- No permissive policies = deny all direct client access by default.
-- All reads and writes go through service_role in server-side API routes.
