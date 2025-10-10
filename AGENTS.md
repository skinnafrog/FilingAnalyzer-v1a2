# Response Format

All responses that use lists, should be numbered or otherwise referencable by user as anchored items.  This includes headings of lists.

# Change Tracking & Management Requirements

## 📋 ERROR_LOG.md Maintenance

**MANDATORY**: After any significant system modification, bug fix, or enhancement:

1. **Update ERROR_LOG.md** with standardized entry format
2. **Assess system restart requirements** using guidelines in CLAUDE.md
3. **Auto-commit productive changes** to GitHub with proper references
4. **Verify ./stop.sh and ./start.sh compatibility** with changes

## 🔄 Automated Workflow

### Before Making Changes:
1. Document planned change in ERROR_LOG.md
2. Assess impact on system components
3. Determine restart requirements

### After Making Changes:
1. Update ERROR_LOG.md with resolution details
2. Test system functionality
3. Execute appropriate restart sequence if needed
4. Auto-push to GitHub with proper commit message

### Restart Decision Matrix:
- **Database models changed**: Full restart (./stop.sh && ./start.sh)
- **API endpoints modified**: Backend restart (docker-compose restart backend)
- **Pipeline changes**: Backend restart
- **Docker configs changed**: Full restart
- **Environment variables**: Service-specific restart
- **Documentation only**: No restart needed

## 📝 GitHub Integration

All productive changes must be automatically pushed with commit messages including:
- Clear change description
- ERROR_LOG.md entry reference
- Impact assessment
- Restart requirements noted
- Claude Code co-authorship tag

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
**EXCEPTION**: ERROR_LOG.md updates are REQUIRED for all significant changes and are not considered proactive documentation.
