// Run with Playwright installed and a local server: node tests/homepage.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:8000');
    await page.locator('#classify').click();
    assert.match(await page.locator('#error').innerText(), /Add a title/);
    await page.locator('[data-example="create"]').click();
    assert.equal(await page.locator('#title').inputValue(), 'Make a birthday song');
    await page.locator('#classify').click();
    await page.locator('#result').waitFor({state: 'visible'});
    assert.equal(await page.locator('.score-row').count(), 7);
    assert.equal(await page.locator('.winner').count(), 1);
    await page.screenshot({path:'/tmp/activity-homepage-desktop.png', fullPage:true});
    await page.locator('#title').fill('A new activity');
    assert.equal(await page.locator('#result').isVisible(), false);
    await page.route('**/predict', route => route.fulfill({status:500, body:'unavailable'}));
    await page.locator('#classify').click();
    await page.locator('#error').waitFor({state:'visible'});
    assert.match(await page.locator('#error').innerText(), /could not complete/);
    assert.equal(await page.locator('#classify').isEnabled(), true);
    await page.unroute('**/predict');
    await page.locator('#reset').click();
    assert.equal(await page.locator('#title').inputValue(), '');
    await page.setViewportSize({width:390,height:844});
    await page.locator('[data-example="practice"]').click();
    await page.locator('#classify').click();
    await page.locator('#result').waitFor({state:'visible'});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path:'/tmp/activity-homepage-mobile.png', fullPage:true});
    assert.deepEqual(errors, []);
    console.log('PASS: real predictions, seven bars, empty input, edits, reset, server errors, mobile overflow, and browser errors.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
