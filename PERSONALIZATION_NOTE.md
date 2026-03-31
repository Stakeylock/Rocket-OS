# Personalization Instructions for Publication

You have a resume file (`jiniresume.pdf`) in this directory. To personalize the research paper for publication with your actual information:

## Steps to Personalize

1. **Update Author Information** in these files:
   - `Autonomous_Rocket_AI_OS_Research_Paper.tex`
   - `Autonomous_Rocket_AI_OS_Research_Paper.md`
   - `references.bib`

2. **Information to Update**:
   - **Author Name**: Replace "Your Name Here" with your actual name
   - **Institution**: Replace "Your Institution Here" with your university/company
   - **Email**: Replace "your.email@institution.edu" with your actual email
   - **GitHub URL**: In references.bib, update `https://github.com/YOUR-USERNAME/Rocket-OS` to your actual GitHub repository

3. **Files That Need Updates**:
   ```
   Autonomous_Rocket_AI_OS_Research_Paper.tex: Lines 20-24
   Autonomous_Rocket_AI_OS_Research_Paper.md: Lines 6-9
   references.bib: Line 56
   ```

4. **Optional**: You may also want to update:
   - Acknowledgments section (if added)
   - Any personal websites or ORCID identifiers

## Verification That Personalization Worked

After updating, you can verify by searching for your name/institution:

```bash
grep -n "Your Name Here" Autonomous_Rocket_AI_OS_Research_Paper.tex
# Should return no results if properly updated

grep -n "Your Institution Here" Autonomous_Rocket_AI_OS_Research_Paper.md
# Should return no results if properly updated
```

## Quick Personalization Commands

If you want to automate updating with your actual details, you could use:

```bash
# Replace with your actual information
ACTUAL_NAME="Your Actual Name"
ACTUAL_INSTITUTION="Your Actual Institution"
ACTUAL_EMAIL="your.email@actual.institution.edu"
GITHUB_USERNAME="your-github-username"

# Update files
sed -i "s/Your Name Here/$ACTUAL_NAME/g" Autonomous_Rocket_AI_OS_Research_Paper.tex
sed -i "s/Your Name Here/$ACTUAL_NAME/g" Autonomous_Rocket_AI_OS_Research_Paper.md
sed -i "s/Your Institution Here/$ACTUAL_INSTITUTION/g" Autonomous_Rocket_AI_OS_Research_Paper.tex
sed -i "s/Your Institution Here/$ACTUAL_INSTITUTION/g" Autonomous_Rocket_AI_OS_Research_Paper.md
sed -i "s/your.email@institution.edu/$ACTUAL_EMAIL/g" Autonomous_Rocket_AI_OS_Research_Paper.tex
sed -i "s/your.email@institution.edu/$ACTUAL_EMAIL/g" Autonomous_Rocket_AI_OS_Research_Paper.md
sed -i "s/Das, Jinitangsu/$ACTUAL_NAME/g" references.bib
sed -i "s/JINITANGSU-DAS/$GITHUB_USERNAME/g" references.bib
```

## Final Verification

Once personalized, rebuild the paper:
```bash
make all  # or follow COMPILATION_GUIDE.md instructions
```

Your research paper will then be ready for submission with your proper attribution.