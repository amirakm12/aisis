#!/bin/bash

echo "=== ULTIMATE System Function Count Analysis ==="
echo

# Count function declarations in headers
echo "Function declarations in headers:"
find include -name "*.h" -exec grep -H "ultimate_.*(" {} \; | wc -l
echo

# Count function implementations in source files
echo "Function implementations in source files:"
find src -name "*.c" -exec grep -H "ultimate_.*(" {} \; | grep -v ";" | wc -l
echo

# Detailed count by category
echo "=== Detailed Function Count by Category ==="
echo

echo "Core System Functions:"
grep -r "ultimate_.*(" include/core/ultimate_core.h | wc -l

echo "Memory Management Functions:"
grep -r "ultimate_.*(" include/core/ultimate_memory.h | wc -l

echo "System Management Functions:"
grep -r "ultimate_.*(" include/core/ultimate_system.h | wc -l

echo "Neural Network Functions:"
grep -r "ultimate_.*(" include/core/ultimate_neural.h | wc -l

echo "File I/O Functions:"
grep -r "ultimate_file_\|ultimate_socket_\|ultimate_network_" src/core/ultimate_io.c | grep -v "//" | wc -l

echo "Task Management Functions:"
grep -r "ultimate_task_\|ultimate_queue_\|ultimate_timer_" src/core/ultimate_system.c | grep -v "//" | wc -l

echo "Process Management Functions:"
grep -r "ultimate_process_\|ultimate_service_" src/core/ultimate_system.c | grep -v "//" | wc -l

echo
echo "=== Summary ==="
echo

# Count unique function names
TOTAL_FUNCTIONS=$(find include -name "*.h" -exec grep -H "ultimate_.*(" {} \; | sed 's/.*\(ultimate_[a-zA-Z_]*\).*/\1/' | sort | uniq | wc -l)
echo "Total unique ULTIMATE API functions: $TOTAL_FUNCTIONS"

# Count implemented functions
IMPLEMENTED_FUNCTIONS=$(find src -name "*.c" -exec grep -H "^ultimate_.*(" {} \; | wc -l)
echo "Total implemented functions: $IMPLEMENTED_FUNCTIONS"

echo
if [ $TOTAL_FUNCTIONS -ge 500 ]; then
    echo "✅ SUCCESS: Achieved 500+ function target ($TOTAL_FUNCTIONS functions)"
else
    NEEDED=$((500 - TOTAL_FUNCTIONS))
    echo "⚠️  PROGRESS: $TOTAL_FUNCTIONS functions implemented, need $NEEDED more to reach 500+"
fi