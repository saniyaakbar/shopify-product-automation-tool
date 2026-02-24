"""Runtime feature flags for development and safe modes."""

# When True, all OpenAI API calls are disabled and the pipeline will
# load existing outputs from disk where possible. This enables a safe
# mode for testing/uploads without consuming OpenAI quota.
DISABLE_OPENAI_CALLS = True
