# HDF5 Documentation Filter Analysis

**Date:** January 13, 2026  
**Source:** HDF5 Official Documentation (support.hdfgroup.org)

## Summary

| Stage | Count | Notes |
|-------|-------|-------|
| Original crawl | 300 pages | Raw from crawler |
| After UniversalFilter | 285 pages | 95% retention |
| After manual TOC removal | 275 pages | 91.7% retention |
| hdf5_filtered (old) | 272 pages | Previous filtered set |

## Filter Configuration

```python
# UniversalFilter settings
min_lines = 30
min_words = 100
max_link_ratio = 0.6

# SKIP_PATTERNS (TIER1 - filename based)
SKIP_PATTERNS = [
    "deprecated", "index", "namespace", "crawl", 
    "search", "sitemap", "404", "error"
]
```

## Pages Removed by UniversalFilter (TIER1 + TIER2)

**15 pages removed automatically:**

| Page | Reason |
|------|--------|
| deprecated | TIER1: Filename pattern |
| doxygen_crawl | TIER1: Filename pattern |
| index | TIER1: Filename pattern |
| namespace_h5 | TIER1: Filename pattern |
| namespacehdf5 | TIER1: Filename pattern |
| e_r_r_o_r_s | TIER1: Filename pattern |
| files | TIER1: Filename pattern |
| f_t_s | TIER1: Filename pattern (full-text search) |
| group_f_m_p_l | TIER2: Too few lines |
| group_m_c_p_l | TIER2: Too few lines |
| h_d_f5_c_o_n_s_t | TIER2: Too few lines |
| h_d_f_a_r_r_a_y | TIER2: Too few lines |
| h_d_f_n_a_t_i_v_e | TIER2: Too few words |
| struct_h5_e_error2_t | TIER2: Too few words |
| downloads_latest | TIER2: Too few words |

## Pages Removed Manually (TOC/Index pages)

**10 pages removed manually after analysis:**

| Page | Words | Links | Reason |
|------|-------|-------|--------|
| command_tools | 126 | 31 | Just links to tool pages |
| cookbook | 169 | 35 | TOC for recipes |
| getting_started | 344 | 33 | TOC for getting started |
| learn_basics | 261 | 40 | TOC for learning basics |
| release_specific_info | 218 | 35 | TOC for release info |
| s_p_e_c | 188 | 31 | TOC for specifications |
| t_n | 216 | 41 | TOC for technical notes |
| view_tools_command | 388 | 38 | TOC for tool commands |
| group_p_d_t | 109 | 26 | Minimal content |
| todo | 120 | 19 | Todo list (not documentation) |

## Pages Kept (that hdf5_filtered didn't have)

**3 additional quality pages in new_web:**

| Page | Words | Links | Content |
|------|-------|-------|---------|
| about | 770 | 44 | Documentation about Documentation |
| a_r_u_g | 387 | 29 | Additional Resources User Guide |
| h_d_f5_examples | 248 | 27 | HDF5 Code Examples by API |

## Filter Improvement Recommendations

### Add to TIER1 SKIP_PATTERNS
```python
# TOC/Navigation pages
"command_tools", "cookbook", "getting_started", "learn_basics",
"release_specific_info", "view_tools_command",
"todo",  # Todo lists

# Single-letter TOCs
"s_p_e_c", "t_n",  # Specifications, Technical Notes TOC
```

### Improve TIER3 (Link Ratio Detection)
Current threshold: `max_link_ratio = 0.6`

Observed patterns for TOC pages:
- High link count (>30) with low word count (<400)
- Link ratio alone may not catch them

**Proposed new rule:**
```python
# If links > 30 AND words < 400 AND link_ratio > 0.08
# Mark as potential TOC page
if links > 30 and words < 400:
    link_ratio = links / words
    if link_ratio > 0.08:  # 8% or more links per word
        return False, "TIER3: TOC page (high links, low content)"
```

### Content Quality Signals

**Good pages have:**
- Code examples
- Technical explanations
- Function signatures
- Parameter descriptions
- Return value documentation

**Bad pages (remove) have:**
- Only navigation links
- Duplicate content
- Empty/placeholder content
- Login/auth pages
- Download lists without documentation

## Key Pages to Preserve

### User Guides (h5_*_u_g)
- h5_u_g - Main User Guide
- h5_d_u_g - Dataset User Guide
- h5_t_u_g - Datatype User Guide
- h5_a_u_g - Attribute User Guide
- h5_f_u_g - File User Guide
- h5_g_u_g - Group User Guide
- h5_p_u_g - Property List User Guide
- h5_s_u_g - Dataspace User Guide
- h5_l_u_g - Link User Guide
- h5_o_u_g - Object User Guide
- h5_r_u_g - Reference User Guide
- h5_z_u_g - Filter User Guide
- h5_e_u_g - Error Handling User Guide
- h5_i_u_g - Identifier User Guide

### Tool Documentation (h5_t_o_o_l_*_u_g)
All 15 tool pages should be preserved:
- h5dump, h5copy, h5diff, h5repack, h5ls
- h5import, h5stat, h5clear, h5debug, h5delete
- h5mkgrp, h5repart, h5jam, h5format_convert, h5watch

### API Reference (group_h5_*)
All API group pages should be preserved for function documentation.

## Additional Manual Cleanup

After automated filtering, manual review found:

### Pages with Very Few Words (<200) - KEPT (valid API docs)
| Page | Words | Status |
|------|-------|--------|
| struct_h5_o_token_t | 131 | KEEP - Valid struct doc |
| h5_f_dstdio_8h | 133 | KEEP - Valid file driver header |
| h5_f_dwindows_8h | 135 | KEEP - Valid file driver header |
| group_h5_f_d | 136 | KEEP - Valid API group |
| h5_m_mpublic_8h | 137 | KEEP - Valid header |
| h5_t_o_o_l_d_t_u_g | 150 | KEEP - h5delete tool (simple command) |
| struct_h5_ih_info_t | 150 | KEEP - Valid struct doc |
| h5_f_dsec2_8h | 161 | KEEP - Valid file driver header |
| h5_t_o_o_l_d_g_u_g | 167 | KEEP - h5debug tool |

### Placeholder Pages Found - REMOVED
| Page | Words | Reason |
|------|-------|--------|
| performance | 162 | Template only ("Problem/Solution/Discussion" placeholders) |

### High Link Ratio Pages - KEPT (valid content)
| Page | Words | Links | Ratio | Status |
|------|-------|-------|-------|--------|
| u_g | 1647 | 343 | 0.20 | KEEP - Main User Guide TOC (valuable) |
| struct_input | 361 | 80 | 0.22 | KEEP - Valid struct for h5import |
| h5version_8h | 1222 | 273 | 0.22 | KEEP - Version info header |

## Final Statistics

- **Total quality pages:** 274
- **Total removed:** 26 (8.7%)
- **Key documentation preserved:** 100%

### Breakdown by Category
| Category | Count |
|----------|-------|
| User Guides | 14 |
| Tool pages | 15 |
| API groups | 32 |
| Struct pages | 11 |
| Header files | ~40 |
| Other docs | ~162 |

## Patterns to Add to Filter

### New TIER1 Patterns (filename-based)
```python
# Add to SKIP_PATTERNS
"command_tools",  # TOC for tools
"cookbook",       # TOC for recipes  
"getting_started", # TOC for getting started
"learn_basics",   # TOC for learning
"release_specific_info",  # TOC for releases
"view_tools_command",     # TOC for tool commands
"s_p_e_c",        # TOC for specifications
"t_n",            # TOC for technical notes
"todo",           # Todo list
"group_p_d_t$",   # Empty parent group
```

### New TIER2 Pattern (content-based)
```python
# Detect placeholder pages
PLACEHOLDER_MARKERS = [
    "The to-the-point solution",
    "A discussion of the fine points",
    "References to related recipes",
]
```
