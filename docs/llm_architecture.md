                          +------------------------+
                          |   Audio Pipeline Record|
                          |  (Transcript + DSP)    |
                          +-----------+------------+
                                      |
                                      v
                        +----------------------------+
                        |      1. Router Node        |
                        | (Evaluates Quality Signals)|
                        +-------------+--------------+
                                      |
                   +------------------+------------------+
                   | (Low Quality /                      | (High Quality /
                   | High Noise)                         | Good Transcript)
                   v                                     v
    +------------------------------+  +------------------------------+
    |   2. Audio QA Agent          |  |  3. PII & Security Agent     |
    |  (Tool: DuckDB SQL Lookup)   |  | (Entity Extraction & Redact) |
    +--------------+---------------+  +--------------+---------------+
                   |                                 |
                   +------------------+--------------+
                                      |
                                      v
                        +----------------------------+
                        | 4. Executive Insight Agent |
                        | (Summary, Action Items)    |
                        +----------------------------+