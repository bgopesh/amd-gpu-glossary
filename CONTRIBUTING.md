# Contributing to AMD GPU Glossary

Thank you for your interest in contributing to the AMD GPU Glossary! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find errors, outdated information, or missing concepts:

1. Check if an issue already exists
2. Create a new issue with a clear description
3. Include references to official AMD documentation when possible

### Suggesting New Content

To suggest new terms or sections:

1. Open an issue describing the proposed addition
2. Explain why the term/concept is important
3. Provide references or sources if available

### Submitting Changes

1. Fork the repository
2. Create a new branch for your changes
3. Make your modifications
4. Ensure consistency with existing content style
5. Submit a pull request with a clear description

## Content Guidelines

### Writing Style

- **Clear and concise**: Explain concepts in straightforward language
- **Technical accuracy**: Verify information against official AMD documentation
- **Consistent terminology**: Use AMD's official terminology
- **Cross-references**: Link related concepts within the glossary

### Format

Each glossary entry should include:

1. **Term name** (as heading)
2. **Brief description** (1-2 sentences)
3. **Key characteristics** (bullet points)
4. **Code examples** (when relevant)
5. **Related terms** (cross-references)

### Example Entry Structure

```markdown
## Term Name

Brief description of the concept in 1-2 sentences.

**Key characteristics:**
- Point 1
- Point 2
- Point 3

**Example (if applicable):**
\`\`\`cpp
// Code example
\`\`\`

**Related:** [Related Term 1](#related-term-1), [Related Term 2](#related-term-2)
```

## Section Organization

- **Device Hardware**: Physical GPU components
- **Device Software**: Programming models and execution
- **Host Software**: Tools, libraries, and APIs
- **Performance**: Optimization and analysis

Place new entries in the most appropriate section.

## GPU Specifications

When adding new GPU specifications to `amd-gpu-specs.json`:

- Use official AMD specifications
- Include all relevant fields
- Maintain consistent JSON structure
- Add source references in comments if possible

## Resources

Official AMD documentation:
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Instinct Product Pages](https://www.amd.com/en/products/accelerators/instinct.html)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD GPU ISA Documentation](https://gpuopen.com/documentation/)

## Code of Conduct

- Be respectful and constructive
- Focus on technical accuracy
- Welcome newcomers and questions
- Assume good intentions

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Ask in your pull request
- Reference this guide

Thank you for helping make the AMD GPU Glossary better!
